import os
import time
import timeit

import click
import emoji
import numpy as np
import pandas as pd
import polars as pl
import polars_bio as pb
import pybedtools
import rich
import yaml
from genomicranges import GenomicRanges
from pygenomics.interval import GenomicBase
from rich.box import MARKDOWN
from rich.console import Console
from rich.table import Table
from rich_tools import table_to_df
from tqdm import tqdm

from logger import logger
from operations import (
    nearest_bioframe,
    nearest_genomicranges,
    nearest_polars_bio,
    nearest_pybedtools,
    nearest_pyranges0,
    nearest_pyranges1,
    overlap_bioframe,
    overlap_genomicranges,
    overlap_polars_bio,
    overlap_pybedtools,
    overlap_pygenomics,
    overlap_pyranges0,
    overlap_pyranges1,
    count_overlaps_bioframe,
    count_overlaps_polars_bio,
    count_overlaps_pyranges0,
    count_overlaps_pyranges1,
    count_overlaps_genomicranges
)
from utils import df2pr0, df2pr1, prepare_datatests


def run_benchmark(
    benchmarks,
    test_cases,
    functions_overlap,
    functions_nearest,
    functions_count_overlaps,
    bech_data_root,
    output_dir,
    baseline,
    export_format,
):
    console = Console(record=True)
    for b in tqdm(benchmarks, desc="Running benchmarks"):
        dataset = b["dataset"]
        operation = b["operation"]
        num_executions = b["num_executions"]
        num_repeats = b["num_repeats"]
        parallel = b["parallel"]
        threads = b["threads"] if parallel else [1]
        input_dataframes = b["input_dataframes"]
        tools = b["tools"]
        if input_dataframes:
            dataframes_io = b["dataframes_io"]
            for d in dataframes_io:
                tools.append(f"polars_bio_{d}")
        print(tools)
        console.log(
            f"## Benchmark {b["name"]} for {operation} with dataset {dataset} \n"
        )
        logger.info(emoji.emojize(f"Running benchmark {b["name"]} :racing_car: "))
        for t in tqdm(b["test-cases"]):
            results = []
            for th in threads:
                pb.ctx.set_option("datafusion.execution.target_partitions", str(th))
                if th != 1:
                    pb.ctx.set_option("datafusion.optimizer.repartition_joins", "true")
                    pb.ctx.set_option(
                        "datafusion.optimizer.repartition_file_scans", "true"
                    )
                    pb.ctx.set_option("datafusion.execution.coalesce_batches", "false")
                else:
                    pb.ctx.set_option("datafusion.optimizer.repartition_joins", "false")
                    pb.ctx.set_option(
                        "datafusion.optimizer.repartition_file_scans", "false"
                    )
                    pb.ctx.set_option("datafusion.execution.coalesce_batches", "false")

                logger.info(
                    emoji.emojize(
                        f"Running benchmark {b["name"]} with {th} threads :gear: "
                    )
                )
                logger.info(
                    emoji.emojize(f"Loading test case {t}... :chequered_flag:  ")
                )
                test = [test for test in test_cases if test["name"] == t][0]
                df_path_1 = f"{bech_data_root}/{dataset}/{test['df_path_1']}"
                df_path_2 = f"{bech_data_root}/{dataset}/{test['df_path_2']}"
                for tool in tqdm(tools):
                    logger.info(
                        emoji.emojize(
                            f"Running benchmark {b["name"]}, test-case {t} for {tool} :hammer_and_wrench:"
                        )
                    )
                    times = None
                    func = None
                    table = None
                    if operation == "overlap":
                        table = [
                            func
                            for func in functions_overlap
                            if f"{operation}_{tool}".startswith(func.__name__)
                        ]
                    elif operation == "nearest":
                        table = [
                            func
                            for func in functions_nearest
                            if f"{operation}_{tool}".startswith(func.__name__)
                        ]
                    elif operation == "count_overlaps":
                        table = [
                            func
                            for func in functions_count_overlaps
                            if f"{operation}_{tool}".startswith(func.__name__)
                        ]
                    else:
                        logger.error(
                            emoji.emojize(f"Operation {operation} not found :no_entry:")
                        )
                        exit(1)
                    if len(table) == 0:
                        logger.warning(
                            emoji.emojize(
                                f"Operation {operation} not found for tool {tool} :no_entry:"
                            )
                        )
                        console.log(
                            f":bulb: Tool {tool} does not support operation {operation}"
                        )
                        continue
                    func = table[0]
                    if tool == "polars_bio":
                        times = timeit.repeat(
                            lambda: func(
                                df_path_1, df_path_2, output_type="polars.LazyFrame"
                            ),
                            repeat=num_repeats,
                            number=num_executions,
                        )
                    elif input_dataframes and tool.startswith("polars_bio_"):
                        input_type = tool.split("_")[2].split(":")[0]
                        output_type = tool.split("_")[2].split(":")[1]
                        df_1 = (
                            pl.read_parquet(df_path_1)
                            if input_type.startswith("polars")
                            else pd.read_parquet(df_path_1.replace("*.parquet", ""))
                        )
                        df_2 = (
                            pl.read_parquet(df_path_2)
                            if input_type.startswith("polars")
                            else pd.read_parquet(df_path_2.replace("*.parquet", ""))
                        )
                        times = timeit.repeat(
                            lambda: func(df_1, df_2, output_type=output_type),
                            repeat=num_repeats,
                            number=num_executions,
                        )
                    elif tool == "bioframe" and th == 1:
                        df_1 = pd.read_parquet(
                            df_path_1.replace("*.parquet", ""), engine="pyarrow"
                        )
                        df_2 = pd.read_parquet(
                            df_path_2.replace("*.parquet", ""), engine="pyarrow"
                        )
                        times = timeit.repeat(
                            lambda: func(df_1, df_2),
                            repeat=num_repeats,
                            number=num_executions,
                        )
                    elif tool == "pyranges0" and th == 1:
                        df_1 = pd.read_parquet(
                            df_path_1.replace("*.parquet", ""), engine="pyarrow"
                        )
                        df_2 = pd.read_parquet(
                            df_path_2.replace("*.parquet", ""), engine="pyarrow"
                        )
                        df_1_pr0 = df2pr0(df_1)
                        df_2_pr0 = df2pr0(df_2)
                        times = timeit.repeat(
                            lambda: func(df_1_pr0, df_2_pr0),
                            repeat=num_repeats,
                            number=num_executions,
                        )
                    elif tool == "pyranges1" and th == 1:
                        df_1 = pd.read_parquet(
                            df_path_1.replace("*.parquet", ""), engine="pyarrow"
                        )
                        df_2 = pd.read_parquet(
                            df_path_2.replace("*.parquet", ""), engine="pyarrow"
                        )
                        df_1_pr1 = df2pr1(df_1)
                        df_2_pr1 = df2pr1(df_2)
                        times = timeit.repeat(
                            lambda: func(df_1_pr1, df_2_pr1),
                            repeat=num_repeats,
                            number=num_executions,
                        )
                    elif tool == "pybedtools" and th == 1:
                        df_1 = pd.read_parquet(
                            df_path_1.replace("*.parquet", ""), engine="pyarrow"
                        ).sort_values(by=["contig", "pos_start", "pos_end"])
                        df_2 = pd.read_parquet(
                            df_path_2.replace("*.parquet", ""), engine="pyarrow"
                        ).sort_values(by=["contig", "pos_start", "pos_end"])
                        df_1_bed = pybedtools.BedTool.from_dataframe(df_1)
                        df_2_bed = pybedtools.BedTool.from_dataframe(df_2)
                        times = timeit.repeat(
                            lambda: func(df_1_bed, df_2_bed),
                            repeat=num_repeats,
                            number=num_executions,
                        )
                    elif tool == "pygenomics" and th == 1:
                        df_1 = pd.read_parquet(
                            df_path_1.replace("*.parquet", ""), engine="pyarrow"
                        )
                        df_2 = pd.read_parquet(
                            df_path_2.replace("*.parquet", ""), engine="pyarrow"
                        )
                        df_1_pg = GenomicBase(
                            [
                                (r.contig, r.pos_start, r.pos_end)
                                for r in df_1.itertuples()
                            ]
                        )
                        df_2_array = df_2.values.tolist()
                        times = timeit.repeat(
                            lambda: func(df_1_pg, df_2_array),
                            repeat=num_repeats,
                            number=num_executions,
                        )
                    elif tool == "genomicranges" and th == 1:
                        df_1 = pd.read_parquet(
                            df_path_1.replace("*.parquet", ""), engine="pyarrow"
                        )
                        df_2 = pd.read_parquet(
                            df_path_2.replace("*.parquet", ""), engine="pyarrow"
                        )
                        df_1_gr = GenomicRanges.from_pandas(
                            df_1.rename(
                                columns={
                                    "contig": "seqnames",
                                    "pos_start": "starts",
                                    "pos_end": "ends",
                                }
                            )
                        )
                        df_2_gr = GenomicRanges.from_pandas(
                            df_2.rename(
                                columns={
                                    "contig": "seqnames",
                                    "pos_start": "starts",
                                    "pos_end": "ends",
                                }
                            )
                        )
                        times = timeit.repeat(
                            lambda: func(df_1_gr, df_2_gr),
                            repeat=num_repeats,
                            number=num_executions,
                        )
                    elif th == 1:
                        logger.error(emoji.emojize(f"Tool {tool} not found :no_entry:"))
                        exit(1)
                    else:
                        continue
                    logger.info(
                        emoji.emojize(
                            f"Finished benchmark for {tool} :check_mark_button:"
                        )
                    )
                    per_run_times = [
                        time / num_executions for time in times
                    ]  # Convert to per-run times
                    results.append(
                        {
                            "name": tool if th == 1 else f"{tool}-{th}",
                            "min": min(per_run_times),
                            "max": max(per_run_times),
                            "mean": np.mean(per_run_times),
                        }
                    )
            if baseline == "fastest":
                baseline_mean = min(result["mean"] for result in results)
            elif baseline == "slowest":
                baseline_mean = max(result["mean"] for result in results)
            else:
                baseline_mean = [
                    result["mean"] for result in results if result["name"] == baseline
                ][0]

            for result in results:
                result["speedup"] = baseline_mean / result["mean"]

                # Create Rich table
            table = Table(
                title=f"Benchmark Results, operation: {operation} dataset: {dataset}, test: {t}",
                box=MARKDOWN,
            )
            table.add_column("Library", justify="left", style="cyan", no_wrap=True)
            table.add_column("Min (s)", justify="right", style="green")
            table.add_column("Max (s)", justify="right", style="green")
            table.add_column("Mean (s)", justify="right", style="green")
            table.add_column("Speedup", justify="right", style="magenta")

            # Add rows to the table
            for result in results:
                table.add_row(
                    result["name"],
                    f"{result['min']:.6f}",
                    f"{result['max']:.6f}",
                    f"{result['mean']:.6f}",
                    f"{result['speedup']:.2f}x",
                )
            rich.print(table)
            console.log(table)
            df = table_to_df(table)
            table_path = f"{output_dir}/{b["name"]}_{t}.{export_format}"
            if export_format == "csv":
                df.to_csv(table_path, index=False)
            console.save_text(
                f"{output_dir}/benchmark_results.md", clear=False, styles=False
            )
        logger.info(
            emoji.emojize(f"Finished benchmark {b["name"]} :check_mark_button:")
        )
    console.print("Benchmark results saved to benchmark_results.md")


@click.command()
@click.option(
    "--bench-config",
    default="conf/benchmark_small.yaml",
    help="Benchmark config file (default: conf/benchmark_small.yaml)",
)
def run(bench_config: str):
    logger.info(f"Using config file: {bench_config}")
    tag_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(emoji.emojize("Starting polars_bio_benchmark :rocket:"))
    # create dir recursively
    output_dir = f"results/{tag_datetime}"
    os.makedirs(output_dir, exist_ok=True)

    REPORT_FILE = f"{output_dir}/benchmark_results.md"
    if os.path.exists(REPORT_FILE):
        logger.info(
            emoji.emojize(
                f"Performance report file {REPORT_FILE} already exist...quiting. Rename/remove it to re-run the benchmark :wastebasket:"
            )
        )
        exit(1)
    BECH_DATA_ROOT = os.getenv("BENCH_DATA_ROOT")
    if not BECH_DATA_ROOT:
        logger.error(
            emoji.emojize("Env variable BENCH_DATA_ROOT is not set :pile_of_poo:")
        )
        exit(1)

    with open("conf/common.yaml", "r") as file:
        config = yaml.safe_load(file)
    logger.info(emoji.emojize("Loaded config. :open_book:"))

    with open(bench_config, "r") as file:
        benchmarks = yaml.safe_load(file)
        logger.info(emoji.emojize("Loaded benchmarks. :open_book:"))
    datasets = config["datasets"]
    test_cases = config["test-cases"]
    benchmarks = benchmarks["benchmarks"]
    baseline = config["benchmark"]["baseline"].lower()
    export_format = config["benchmark"]["export"]["format"].lower()
    functions_overlap = [
        overlap_polars_bio,
        overlap_bioframe,
        overlap_pyranges0,
        overlap_pyranges1,
        overlap_pybedtools,
        overlap_pygenomics,
        overlap_genomicranges,
    ]

    functions_nearest = [
        nearest_polars_bio,
        nearest_bioframe,
        nearest_pyranges0,
        nearest_pyranges1,
        nearest_pybedtools,
        nearest_genomicranges,
    ]
    functions_count_overlaps = [
        count_overlaps_polars_bio,
        count_overlaps_bioframe,
        count_overlaps_pyranges0,
        count_overlaps_pyranges1,
        count_overlaps_genomicranges,
    ]


    prepare_datatests(datasets, BECH_DATA_ROOT)

    run_benchmark(
        benchmarks,
        test_cases,
        functions_overlap,
        functions_nearest,
        functions_count_overlaps,
        BECH_DATA_ROOT,
        output_dir,
        baseline,
        export_format,
    )
    logger.info(emoji.emojize("Finished polars_bio_benchmark :check_mark_button:"))


if __name__ == "__main__":
    run()
