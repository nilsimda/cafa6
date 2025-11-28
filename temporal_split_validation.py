import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import requests
    import json
    import re
    return json, pl, re, requests


@app.cell
def _(requests):
    valid_evidence_codes = {
        "EXP", "IDA", "IPI", "IMP", "IGI", "IEP",  # Experimental
        "HTP", "HDA", "HMP", "HGI", "HEP",         # High-throughput
        "TAS",                                     # Traceable Author Statement
        "IC"                                       # Inferred by Curator
    }

    query_ls = []
    query_ls.append(" OR ".join([f"GO_{ec}:*" for ec in valid_evidence_codes])) 
    query_ls.append("taxonomy_id:")
    query_ls.append(f"date_modified:[* TO 2025-06-18]")
    query_ls.append("reviewed:true") # Swiss-Prot only

    full_query = " AND ".join(f"({subquery})" for subquery in query_ls)
    api_url = "https://rest.uniprot.org/uniprotkb/search"
    # request with size=0 to fetch metadata only (faster)
    response = requests.get(
        api_url, 
        params={
            "query": full_query, 
            "size": 0, 
            "format": "json"
        }
    )
    response.raise_for_status()

    total_count = response.headers.get("x-total-results")
    int(total_count)
    return full_query, valid_evidence_codes


@app.cell
def _(pl):
    train_terms = pl.read_csv("dataset/Train/train_terms.tsv", separator="\t")
    train_terms.head(10)
    return (train_terms,)


@app.cell
def _(full_query, json, re, requests):
    def _fetch_uniprot_results(query_str):
        """Fetches all results from UniProt via pagination."""
        base_url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query_str,
            "size": 500,
            "format": "json",
            "fields": "accession,sequence,go,organism_id"
        }
    
        results = []
        next_link = None
    
        while True:
            request_url = next_link if next_link else base_url
            # If using next_link, parameters are embedded in the URL
            request_params = None if next_link else params
        
            resp = requests.get(request_url, params=request_params)
            resp.raise_for_status()
        
            data = resp.json()
            if "results" in data:
                results.extend(data["results"])
            
            link_header = resp.headers.get("Link")
            if link_header:
                match = re.search(r'<(.+)>; rel="next"', link_header)
                if match:
                    next_link = match.group(1)
                else:
                    break
            else:
                break
            
        return results

    # 1. Fetch raw data
    print(f"Fetching data for validation set construction...")
    raw_data = _fetch_uniprot_results(full_query)

    with open("uniprot_valid_results.json", "w") as out:
        out.write(json.dumps(raw_data))
    return (raw_data,)


@app.cell
def _(raw_data):
    raw_data[2] 
    return


@app.cell
def _(pl):
    test_taxon_df = pl.read_csv("dataset/Test/testsuperset-taxon-list.tsv", separator="\t")
    test_taxon_df
    return (test_taxon_df,)


@app.cell
def _(pl, raw_data, test_taxon_df, train_terms, valid_evidence_codes):
    # Function to parse the UniProt JSON structure
    def extract_go_terms(data, valid_codes, valid_taxons):
        extracted = []
        for entry in data:
            accession = entry.get("primaryAccession")
            organism = entry.get("organism")
            if not accession or not organism:
                continue

            if not int(organism.get("taxonId")) in valid_taxons:
                continue
            
            # Loop through cross-references to find GO terms
            xrefs = entry.get("uniProtKBCrossReferences", [])
            for xref in xrefs:
                if xref.get("database") == "GO":
                    props = {p["key"]: p["value"] for p in xref.get("properties", [])}
                
                    # Parse Evidence Code (e.g., "IDA:PubMed:...")
                    ev_type = props.get("GoEvidenceType", "")
                    current_code = ev_type.split(":")[0] if ":" in ev_type else ev_type
                
                    if current_code in valid_codes:
                        go_term = xref.get("id")
                        term_def = props.get("GoTerm", "")
                        # Aspect is typically the first letter (C, F, P)
                        aspect = term_def[0] if term_def else None
                    
                        if aspect in ("C", "F", "P"):
                            extracted.append({
                                "EntryID": accession,
                                "term": go_term,
                                "aspect": aspect
                            })
        return extracted

    valid_taxons = set(test_taxon_df["ID"].unique().to_list())
    # Parse the raw data collected in previous cells
    _new_go_pairs = extract_go_terms(raw_data, valid_evidence_codes, valid_taxons)
    _new_terms_df = pl.DataFrame(_new_go_pairs, schema={"EntryID": pl.String, "term": pl.String, "aspect": pl.String})

    if _new_terms_df.height > 0:
        # Deduplicate the collected entries
        _new_terms_df = _new_terms_df.unique()

        # Identify terms present in new data but missing in train_terms (anti-join)
        # Using EntryID and term as the composite key
        additional_terms_df = _new_terms_df.join(
            train_terms, 
            on=["EntryID", "term"], 
            how="anti"
        )
    else:
        additional_terms_df = pl.DataFrame(schema={"EntryID": pl.String, "term": pl.String, "aspect": pl.String})

    print(f"Found {additional_terms_df.height} new valid protein-GO pair(s) not in training set.")
    additional_terms_df
    return (additional_terms_df,)


@app.cell
def _(pl):
    train_df = pl.read_parquet("dataset/Train/trainset_with_onehot_taxas.parquet")
    train_df
    return


@app.cell
def _():
    return


@app.cell
def _(additional_terms_df):
    additional_terms_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
