## Summary
 This repository contains the code required for the analysis of the manuscripts:
 * [Understanding soaring coronavirus cases and the effect of contagion policies in the UK](https://doi.org/10.3390/vaccines9070735).
 * [More than a year after the onset of the CoVid-19 pandemic in the UK: lessons learned from a minimalistic model capturing essential aspects including social awareness and policy making](https://doi.org/10.1101/2021.04.15.21255510)

The running of the script `uk.py` will produce the figures showed in the manuscript.
These figures will be saved in PDF format inside the [figs](figs) folder.

![](slides/sir-covid-uk.png "Effect of lockdown and vaccination")

## Brief description
The script here presented runs the following steps:

1. Integrate a modified SIR model over time by using `scipy.integrate.odeint`. This solver integrates ordinary differential equations using `LSODA` from the
    `FORTRAN` library `odepack`.
2. Fits the parameters of a modified SIR model by using `scipy.optimize.curve_fit`, which is a friendly wrapper for non-linear least squares, over the training dataset
3. Integrates the modified SIR model over time by using the parameters coming from (2). This is contrasted against the testing dataset (data as of September 2020) 
 
