# voiscnd

This repository contains the official replication code and data primitives for the manuscript:

> **"The Value of Information in Supply Chain Network Design: A Three-Stage Stochastic Programming Approach"**

---

## Computational Experiments & Replication

The core numerical experiments, Sample Average Approximation (SAA) macro-runs, and case studies are structured as individual Python scripts (`voi*.py`). 

### Running on a High-Performance Computing (HPC) Cluster

To execute the full benchmarking suite on a cluster running a workload manager like SLURM, you can use the provided `run_single.sh` batch script. Run the following command in your terminal to create the necessary logging directories and submit each experiment as an independent batch job:

```bash
# Create a logs directory to keep output streams clean
mkdir -p logs

# Loop through and submit each Python file as a separate batch job
for file in voi*.py; do
    sbatch run_single.sh "$file"
done
```

## Analyzing Results

Once the computational scripts have completed and saved their data primitives, you can run the interactive visualization and sensitivity analysis notebooks (`*.ipynb`) individually using an IDE such as **VS Code** or **JupyterLab**.

## Citation

If you use this code, data, or framework in your research, or if you plan to extend the methodology, please cite the foundational research paper using the following BibTeX entry:

```bibtex
@article{saragih2024pivotal,
  title={The Value of Information in Supply Chain Network Design: A Three-Stage Stochastic Programming Approach},
  author={Saragih, Austin and Janjevic, Milena and Winkenbach, Matthias and Montibeller, Gilberto},
  journal={MIT Center for Transportation \& Logistics Research Paper},
  number={2024/012},
  year={2024}
}
```
