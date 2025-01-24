import itertools

import bioframe as bf
import polars_bio as pb

columns = ("contig", "pos_start", "pos_end")


def nearest_bioframe(df_1, df_2):
    len(bf.closest(df_1, df_2, cols1=columns, cols2=columns))


def nearest_polars_bio(df_path_1, df_path_2, output_type):
    if output_type == "polars.LazyFrame":
        len(
            pb.nearest(
                df_path_1,
                df_path_2,
                cols1=columns,
                cols2=columns,
                output_type=output_type,
            ).collect()
        )
    else:
        len(
            pb.nearest(
                df_path_1,
                df_path_2,
                cols1=columns,
                cols2=columns,
                output_type=output_type,
            )
        )


def nearest_pyranges0(df_1_pr0, df_2_pr0):
    len(df_1_pr0.nearest(df_2_pr0))


def nearest_pyranges1(df_1_pr1, df_2_pr1):
    len(df_1_pr1.nearest(df_2_pr1))


def nearest_pybedtools(df_1_bed, df_2_bed):
    len(df_1_bed.closest(df_2_bed, s=False, t="first"))


def nearest_genomicranges(df_1, df_2):
    len(df_1.nearest(df_2, ignore_strand=True, select="arbitrary"))


def overlap_bioframe(df_1, df_2):
    len(bf.overlap(df_1, df_2, cols1=columns, cols2=columns, how="inner"))


def overlap_polars_bio(df_path_1, df_path_2, output_type):
    if output_type == "polars.LazyFrame":
        len(pb.overlap(df_path_1, df_path_2, cols1=columns, cols2=columns).collect())
    else:
        len(
            pb.overlap(
                df_path_1,
                df_path_2,
                cols1=columns,
                cols2=columns,
                output_type=output_type,
            )
        )


def overlap_pyranges0(df_1_pr0, df_2_pr0):
    len(df_1_pr0.join(df_2_pr0))


def overlap_pyranges1(df_1_pr1, df_2_pr1):
    len(df_1_pr1.join_ranges(df_2_pr1))


def overlap_pybedtools(df_1_bed, df_2_bed):
    len(df_1_bed.intersect(df_2_bed, wa=True, wb=True))


def overlap_pygenomics(df_1_pg, df_2_array):
    len(
        list(
            itertools.chain.from_iterable(
                [df_1_pg.find_all((r[0], r[1], r[2])) for r in df_2_array]
            )
        )
    )


def overlap_genomicranges(df_1, df_2):
    len(df_1.find_overlaps(df_2, ignore_strand=True, query_type="any"))



def count_overlaps_bioframe(df_1, df_2):
    len(bf.count_overlaps(df_1, df_2, cols1=columns, cols2=columns))


def count_overlaps_polars_bio(df_path_1, df_path_2, output_type):
    go_naive = False
    if output_type == "polars.LazyFrame":
        len(
            pb.count_overlaps(
                df_path_1,
                df_path_2,
                cols1=columns,
                cols2=columns,
                output_type=output_type,
                naive_query=go_naive,
            ).collect()
        )
    else:
        len(
            pb.count_overlaps(
                df_path_1,
                df_path_2,
                cols1=columns,
                cols2=columns,
                output_type=output_type,
                naive_query=go_naive,
            )
        )


def count_overlaps_pyranges0(df_1_pr0, df_2_pr0):
    len(df_1_pr0.count_overlaps(df_2_pr0))


def count_overlaps_pyranges1(df_1_pr1, df_2_pr1):
    len(df_1_pr1.count_overlaps(df_2_pr1))

'''
def count_overlaps_pybedtools(df_1_bed, df_2_bed):
    len(df_1_bed.closest(df_2_bed, s=False, t="first"))
'''

def count_overlaps_genomicranges(df_1, df_2):
    len(df_1.count_overlaps(df_2))

