# Project Title
---
**Description:** A brief description of what this project does and who it's for

# Installation

# Quick Start

# Authors




# Apendices
## Appendix A - Project Structure
This project's file structure was adapted on:
1. Noble WS (2009) [A Quick Guide to Organizing Computational Biology Projects](https://doi.org/10.1371/journal.pcbi.1000424). PLOS Computational Biology 5(7)
2. [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/#cookiecutter-data-science)

### Directory Structure
---

Legend:  
**Folders are written in bold**  
*Sub-Folders ares written in itallic*  
~~(Sub-)Folders not applicable for this project~~  
`Files are written in codeblock`  
Descriptions are written in plain text  on the RHS of an '<-'

---  
    
- **Data**
  - *External*       <- Data from third party sources.
  - *Interim*        <- Intermediate data that has undergone any processing.
  - *Processed*      <- The final, canonical data set(s) for visualisation, analysis and/or modeling.
  - *Raw*            <- The original, immutable data dump.  
- **Models**         <- Trained and serialised models and model summaries.
- **Notebooks**      <- Jupyter notebooks.
- **Resources**      <- Data dictionaries, manuals, and all other explanatory material.
- **Reports**        <- Generated analysis report.
  - *Figures*        <- Generated graphics and figures.
- **packrat**        <- Dependency management system for R.
  - *lib*            <- Private package library for this project.
  - *src*            <- Source packages of all the dependencies recorded in packrat for this project.
  - `packrat.lock`   <- Lists the precise package versions that were used to satisfy dependencies.
  - `packrat.opts`   <- Project-specific packrat options.
- **Src**            <- Source code files for this project.
  - *R*              <- R scripts.
  - *Python*         <- Python scripts.
- `README.md`        <- This file. 
- `Requirements.txt` <- Python requirement file(s) to reproduce the original analysis environment.
- `Main_R`           <- Master file for R Analysis.
- `Main_Py`          <- Master file for Python Analysis.
- `Project.Rproj`    <- R project file.
- `.Rprofile`        <- (Hidden file) Directs R to use the private package library.
- `.here`            <- (Hidden file) Anchor file for pyHere module.
