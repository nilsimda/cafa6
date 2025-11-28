from Bio import SeqIO
import polars as pl

def create_test_df(fasta_path: str = "dataset/Test/testsuperset.fasta") -> pl.DataFrame:
    """Create a dataframe with the protein ids, sequences and taxon_ids from the test superset fasta file."""
    data = [
        (*record.description.split(" ", 1), str(record.seq))
        for record in SeqIO.parse(fasta_path, "fasta")
    ]

    return pl.DataFrame(
        data,
        schema=[
            ("protein_id", pl.Utf8),
            ("taxon_id", pl.Int64),
            ("sequence", pl.Utf8),
        ],
        orient="row"
    )

def create_train_df(fasta_path: str = "dataset/Train/train_sequences.fasta", taxon_file: str = "dataset/Train/train_taxonomy.tsv") -> pl.DataFrame:
    """Create a dataframe from the train fasta file.

    Columns:
        - db: The database the protein was found in (always sp).
        - protein_id: The uniprot accession id.
        - gene_name: The name of the gene the protein is associated with.
        - sequence: The amino acid sequence of the protein.
        - taxon_id: The taxon id of the organism the protein is associated with.
    """
    data = [
        (*record.id.split("|", 2), str(record.seq))
        for record in SeqIO.parse(fasta_path, "fasta")
    ]

    train_df = pl.DataFrame(
        data, 
        schema=["db", "protein_id", "gene_name", "sequence"],
        orient="row"
    )

    train_taxons_df = pl.read_csv(taxon_file, separator="\t", has_header=False).rename({"column_1": "protein_id", "column_2": "taxon_id"})

    return train_df.join(train_taxons_df, on="protein_id")
