---
title: "Data"
---

All data of this study can be found on the [Github repository](https://github.com/edwinreuvers/rat-gm-ampo). The `data/` folder is divided into three main parts:

* **`dataExp/`** → Experimentally measured data
* **`simsExp/`** → Hill-type MTC model predictions of these experiments (with experimentally measured MTC length and muscle stimulation as input to the MTC model) 
* **`simsCV/`** → Hill-type MTC model predictions for a broad range of stretch-shortening cycles where MTC velocity is constant during the shortening and lengthening phase
* **`simsOC/`** → Hill-type MTC model predictions for a broad range of stretch-shortening cycles, with no constraint on MTC length over time

## `dataExp/` – Experimental data
This folder contains all experimentally measured data from the three rats included in the study.

### Level 2 - Rat folder

* **`GMe1/`**, **`GMe2/`**, **`GMe3/`** – Data from individual rats.

#### Level 3- Experimental protocol

Each rat folder contains data from the following experimental protocols:

* **`ISOM/`** – Isometric contractions
* **`QR/`** – Quick-release contractions
* **`SR/`** – Step-ramp contractions
* **`SSC_PA/`** – Stretch–shortening cycles with a 4 mm MTC length excursion
* **`SSC_PB/`** – Stretch–shortening cycles with an 8 mm MTC length excursion

## `simsExp/` – Model predictions: experimental data

This folder contains Hill-type MTC model predictions corresponding to the experimentally measured protocols in `dataExp/`. Simulations were performed using the measured MTC length and stimulation over time as inputs for each individual experiment.

The folder structure mirrors that of `dataExp/`.

## `simsCV/` – Model predictions: SSCs constant MTC velocity

This folder contains model predictions for a large set of SSCs in which MTC velocity is constant during both the lengthening and shortening phases.

Files follow the naming convention:

`GMe<X1>_cf<X2>Hz_fts<X3>_mle<X4>`

where:

* **`<X1>`** – Rat ID
* **`<X2>`** – Cycle frequency (Hz)
* **`<X3>`** – FTS value
* **`<X4>`** – MTC length excursion (mm)

Examples:

* `GMe2_cf1.5Hz_fts0.65_mle5.0mm`

  * Rat: 2
  * Cycle frequency: 1.5 Hz
  * FTS: 0.65
  * MTC length excursion: 5.0 mm

* `GMe3_cf3.5Hz_ftsOpt_mleOpt`

  * Rat: 3
  * Cycle frequency: 3.5 Hz
  * FTS and MTC length excursion optimised to maximise attainable AMPO
  
## `simsOC/` – Model predictions: unconstrained SSCs

This folder contains model predictions for SSCs in which MTC length and velocity trajectories are not constrained. For each simulation, one SSC parameter is fixed while all remaining parameters are optimised to maximise attainable AMPO.

Files follow the naming convention:

`GMe<X1>_cf<X2>Hz_fts<X3>_mle<X4>`

where:

* **`<X1>`** – Rat ID
* **`<X2>`** – Cycle frequency (Hz)
* **`<X3>`** – FTS value
* **`<X4>`** – MTC length excursion (mm)

Example:

* `GMe2_cf1.5Hz_ftsOpt_mleOpt`

  * Rat: 2
  * Cycle frequency: 1.5 Hz
  * FTS and MTC length excursion optimised to maximise attainable AMPO

## Folder tree
```plaintext
data/
├── GMe1/                         # Rat 1
│   ├── dataExp/                  # Experimental data
│   │   ├── ISOM/
│   │   ├── QR/
│   │   ├── SR/
│   │   ├── SSC_PA/
│   │   └── SSC_PB/
│   ├── simsExp/                  # Simulations of experimental protocols
│   │   ├── ISOM/
│   │   ├── QR/
│   │   ├── SR/
│   │   ├── SSC_PA/
│   │   └── SSC_PB/
│   ├── simsCV/                   # SSC simulations with constant MTC velocity
│   └── simsOC/                   # SSC simulations with unconstrained MTC length/velocity over time
├── GMe2/
│   └── same structure as GMe1
└── GMe3/
    └── same structure as GMe1
```