# Fairness-Enhancing Classification Methods For Non-Binary Sensitive Features - How to Fairly Detect Leakages in Water Distribution Networks
This repository contains the implementation of the methods proposed in the paper ["Fairness-Enhancing Classification Methods For Non-Binary Sensitive Features - How to Fairly Detect Leakages in Water Distribution Networks"](Paper.pdf) by Janine Strotherm, Inaam Ashraf and Barbara Hammer.
This paper is an extended version of the paper ["Fairness-Enhancing Ensemble Classification in Water Distribution Networks"](https://github.com/jstrotherm/FairnessInWDNs/blob/main/Paper.pdf) by Janine Strotherm and Barbara Hammer.

## Abstract
Especially if AI-supported decisions affect the society, the fairness of such AI-based methodologies constitutes an important area of research. In this contribution, we investigate the applications of AI to the socioeconomically relevant infrastructure of water distribution systems (WDSs). We propose an appropriate definition of protected groups in WDSs and generalized definitions of group fairness that provably coincide with existing definitions in their specific settings. We demonstrate that typical methods for the detection of leakages in WDSs are unfair in this sense. Further, we thus propose a general fairness-enhancing framework as an extension of the specific leakage detection pipeline, but also for an arbitrary learning scheme, to increase the fairness of the AI-based algorithm. Finally, we evaluate and compare several specific instantiations of this framework on a toy and on a realistic WDS to show their utility.

## Details
The implementation of the proposed methods can be found in the `Implementation` folder. 

The data required for these methods are stored or can be generated using the `2_DataGeneration` subfolder:
-   The subfolder `2_DataGeneration/Hanoi` holds the data associated with the Hanoi network.
    It is the same data as used in [this previous work](https://github.com/jstrotherm/FairnessInWDNs/blob/main/Paper.pdf). 
    For the data generation, we refer to [this previous repository](https://github.com/jstrotherm/FairnessInWDNs). 
    In this repository, we only store the resulting excel files.
-   The subfolder `2_DataGeneration/L-Town` is a modified version of [this previous repository](https://github.com/HammerLabML/GCNs_for_WDS). 
    Running the `gen_scenario_leakages.py` script generates different leakage scenarios. 
    In ordner to run the script, it is required to 
    a) download [this .inp file](https://github.com/KIOS-Research/BattLeDIM/blob/master/Dataset%20Generator/L-TOWN_v2_Real.inp) and store it in the `2_DataGeneration/L-Town/networks/L-Town/Real` subfolder and 
    b) train a model as specified in [this previous repository](https://github.com/HammerLabML/GCNs_for_WDS) and store it as `2_DataGeneration/L-Town/trained_models/model_L-TOWN_2880_45_1.pt`. 
    The files are not stored in this repository due to their sizes. 
    Consecutively, running the `get_scenario_residuals.py` script generates the data associated with the L-Town network and stores it in csv files. 
    Finally, running the `get_scenario_residuals.ipynb` notebook generates another csv file with network information.
    The csv files are not stored in this repository due to their sizes. 
-   The excel and csv files are in turn used in the `3_DataUsage` subfolder.

The methods themselves can be used using the `3_DataUsage` subfolder:
-   In the `FairnessExploration_Hanoi_extended.ipynb` and in the `FairnessExploration_L-Town.ipynb` notebook, the proposed approaches are implemented.

## Requirements
All requirements for the whole project are listed in the `Implementation/requirements.txt` file.

## How To Cite
tba