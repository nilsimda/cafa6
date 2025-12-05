import marimo

__generated_with = "0.18.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from Bio import SeqIO
    from Bio.SwissProt import parse
    from utils.data import create_test_df, create_train_df
    from tqdm import tqdm
    import polars as pl
    return create_test_df, create_train_df, parse, pl, tqdm


@app.cell
def _(create_test_df, create_train_df):
    SWISSPROT_FILE = "swissprot_release/uniprot_sprot.dat"
    test_df, train_df = create_test_df(), create_train_df()
    return SWISSPROT_FILE, test_df, train_df


@app.cell
def _(test_df, train_df):
    test_ids = set(test_df["protein_id"].to_list())
    train_ids = set(train_df["protein_id"].to_list())
    return (test_ids,)


@app.cell
def _(SWISSPROT_FILE, parse, pl, tqdm):
    def create_nlp_df(prot_ids):
        swissprot_iterator = parse(SWISSPROT_FILE)
        data = []
        for record in tqdm(swissprot_iterator):
            intersection = set(record.accessions).intersection(prot_ids)
            if len(intersection) != 0:
                protein_id = intersection.pop()
            
                data.append({
                    "protein_id": protein_id,
                    "description": record.description,
                    "comments": record.comments,
                    "reference_positions": [str(ref.positions) for ref in record.references],
                    "reference_titles": [ref.title for ref in record.references],
                    "reference_comments": [str(ref.comments) for ref in record.references],
                })
        
        return pl.DataFrame(data)
    return (create_nlp_df,)


@app.cell
def _(create_nlp_df, test_ids):
    test_nlp_df = create_nlp_df(test_ids)
    test_nlp_df
    return (test_nlp_df,)


@app.cell
def _(test_nlp_df):
    test_nlp_df.write_parquet("dataset/test_nlp_data.parquet")
    return


@app.cell
def _():
    "dataset/Test/"
    return


if __name__ == "__main__":
    app.run()
