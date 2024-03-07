#!/bin/bash

set -e

sbatch slurm_axolotl.sh all
sbatch slurm_axolotl.sh light_pubmed_general
sbatch slurm_axolotl.sh light_pubmed_heavy_clinical
sbatch slurm_axolotl.sh pubmed
sbatch slurm_axolotl.sh pubmed_clinical
sbatch slurm_axolotl.sh pubmed_code
sbatch slurm_axolotl.sh pubmed_general
sbatch slurm_axolotl.sh pubmed_heavy_code
sbatch slurm_axolotl.sh pubmed_heavy_reference
sbatch slurm_axolotl.sh pubmed_light_general
sbatch slurm_axolotl.sh pubmed_reference
