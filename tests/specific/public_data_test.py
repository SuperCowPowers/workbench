from workbench.api import PublicData

if __name__ == "__main__":

    # Grab the public LogP data
    pub_data = PublicData()
    df = pub_data.get("logp/logp_all")
    print(df.head())
