
# Workbench Tests
**Important:** In order to run the tests you'll need to run the scripts in the `/create_test_artifacts` directory to create the AWS ML Pipeline objects that get tested. 

At a minimum run these two:

```
cd workbench/test/create_test_artifacts
python create_basic_test_artifacts.py
python create_wine_artifacts.py
```

## Running the tests
Okay the testing ML Pipeline object only need to be created once, so after that you can run the Workbench test suite. In the top level directory of the Workbench repository you can simply run

```
tox
```

## Questions?
The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8) 
<img align="right" src="../docs/images/scp.png" width="180">
