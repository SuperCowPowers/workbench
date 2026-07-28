from workbench.api import PublicData

if __name__ == "__main__":

    # Grab the public LogP data
    pub_data = PublicData()
    df = pub_data.get("comp_chem/logp/logp_all")
    print(df.head())

    # Every published dataset carries a description
    undescribed = [name for name in pub_data.list() if name != "descriptions" and pub_data.describe(name) is None]
    print(f"Undescribed datasets: {undescribed}")
