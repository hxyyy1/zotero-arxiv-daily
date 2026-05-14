from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
import random
import time
from queue import Empty
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180

# arXiv API 防限流参数，可用环境变量覆盖
ARXIV_API_BATCH_SIZE = int(os.getenv("ARXIV_API_BATCH_SIZE", "10"))
ARXIV_API_DELAY_SECONDS = float(os.getenv("ARXIV_API_DELAY_SECONDS", "60"))
ARXIV_CLIENT_NUM_RETRIES = int(os.getenv("ARXIV_CLIENT_NUM_RETRIES", "2"))

ARXIV_BACKOFF_MAX_ATTEMPTS = int(os.getenv("ARXIV_BACKOFF_MAX_ATTEMPTS", "8"))
ARXIV_BACKOFF_BASE_SECONDS = float(os.getenv("ARXIV_BACKOFF_BASE_SECONDS", "30"))
ARXIV_BACKOFF_MAX_SECONDS = float(os.getenv("ARXIV_BACKOFF_MAX_SECONDS", "300"))


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _is_retryable_arxiv_error(exc: Exception) -> bool:
    """
    arXiv API 常见临时错误：
    - HTTP 429: Too Many Requests，限流
    - HTTP 503: Service Unavailable，服务临时不可用
    """
    text = str(exc)
    return (
        "HTTP 429" in text
        or "HTTP 503" in text
        or "Too Many Requests" in text
        or "Service Unavailable" in text
    )


def _sleep_before_retry(attempt: int, exc: Exception, paper_ids: list[str]) -> None:
    sleep_seconds = min(
        ARXIV_BACKOFF_MAX_SECONDS,
        ARXIV_BACKOFF_BASE_SECONDS * (2 ** attempt),
    )

    # 加一点随机抖动，避免 GitHub Actions 多个任务同时重试
    sleep_seconds += random.uniform(0, 5)

    logger.warning(
        "arXiv API temporarily failed for batch {}. "
        "Attempt {}/{}. Error: {}. Sleeping {:.1f}s before retrying.",
        paper_ids,
        attempt + 1,
        ARXIV_BACKOFF_MAX_ATTEMPTS,
        exc,
        sleep_seconds,
    )
    time.sleep(sleep_seconds)


def _fetch_arxiv_batch_with_backoff(
    client: arxiv.Client,
    paper_ids: list[str],
) -> list[ArxivResult]:
    search = arxiv.Search(id_list=paper_ids)

    for attempt in range(ARXIV_BACKOFF_MAX_ATTEMPTS):
        try:
            return list(client.results(search))
        except arxiv.HTTPError as exc:
            if not _is_retryable_arxiv_error(exc):
                raise

            if attempt == ARXIV_BACKOFF_MAX_ATTEMPTS - 1:
                logger.error(
                    "arXiv API failed after {} attempts for batch {}: {}",
                    ARXIV_BACKOFF_MAX_ATTEMPTS,
                    paper_ids,
                    exc,
                )
                raise

            _sleep_before_retry(attempt, exc, paper_ids)

    # 理论上不会走到这里，只是为了类型检查
    return []


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        client = arxiv.Client(
            num_retries=ARXIV_CLIENT_NUM_RETRIES,
            delay_seconds=ARXIV_API_DELAY_SECONDS,
            page_size=ARXIV_API_BATCH_SIZE,
        )

        query = "+".join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)

        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if "Feed error for query" in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")

        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}

        all_paper_ids = [
            entry.id.removeprefix("oai:arXiv.org:")
            for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_announce_types
        ]

        # 去重但保持顺序，避免重复请求同一篇论文
        all_paper_ids = list(dict.fromkeys(all_paper_ids))

        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        raw_papers: list[ArxivResult] = []

        # Get full information of each paper from arxiv api
        with tqdm(total=len(all_paper_ids)) as bar:
            for i in range(0, len(all_paper_ids), ARXIV_API_BATCH_SIZE):
                paper_ids = all_paper_ids[i : i + ARXIV_API_BATCH_SIZE]

                batch = _fetch_arxiv_batch_with_backoff(client, paper_ids)
                raw_papers.extend(batch)

                # 这里按请求过的 paper id 数更新进度，而不是按返回结果数。
                # 如果 arXiv 少返回某条记录，进度条也不会卡住。
                bar.update(len(paper_ids))

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url

        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None

    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None

    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
