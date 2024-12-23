# app.py
import argparse
from fastapi import FastAPI, Path, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
from uvicorn import run
import pathlib
from microsearch.engine import SearchEngine
import multiprocessing

script_dir = pathlib.Path(__file__).resolve().parent
templates_path = script_dir / "templates"
static_path = script_dir / "static"

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def init_search_engine():
    return SearchEngine()


app = FastAPI()
templates = Jinja2Templates(directory=str(templates_path))
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


engine = None


@app.on_event("startup")
async def startup_event():
    global engine
    engine = init_search_engine()


@app.get("/", response_class=HTMLResponse)
async def search(request: Request):
    posts = engine.posts if engine else []
    return templates.TemplateResponse(
        "search.html", {"request": request, "posts": posts}
    )


@app.get("/results/{query}", response_class=HTMLResponse)
async def search_results(request: Request, query: str = Path(...)):
    if not engine:
        return templates.TemplateResponse(
            "error.html", {"request": request, "error": "Search engine not initialized"}
        )
    results = engine.search(query)
    results = get_top_urls(results, n=5)
    return templates.TemplateResponse(
        "results.html", {"request": request, "results": results, "query": query}
    )


def get_top_urls(scores_dict: dict, n: int):
    sorted_urls = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    top_n_urls = sorted_urls[:n]
    return dict(top_n_urls)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    return parser.parse_args()


if __name__ == "__main__":
    # Add multiprocessing support
    multiprocessing.freeze_support()

    args = parse_args()

    try:
        data = pd.read_parquet(args.data_path)
        content = list(zip(data["URL"].values, data["content"].values))


        engine = init_search_engine()

        #try and catch implemented, for debugging purposes...
        try:
            engine.bulk_index(content, chunk_size=1000)
        except Exception as e:
            logger.error(f"Error during bulk indexing: {e}")
            raise

        # Run the FastAPI app
        run(app, host="127.0.0.1", port=8000)

    except Exception as e:
        logger.error(f"Error starting application: {e}")
        raise