import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    identifiers = [c for c in anon_df.columns if c in aux_df.columns and c not in ['anon_id', 'name']]
    merged = anon_df.merge(aux_df, on=identifiers)
    matches = merged.drop_duplicates(subset=['anon_id'], keep=False)
    matches = matches.rename(columns={'name': 'matched_name'})
    return matches[['anon_id', 'matched_name']]

def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0
    return len(matches_df) / len(anon_df)