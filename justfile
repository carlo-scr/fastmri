# FA-KGD MRI Reconstruction — Project Commands
# Usage: just <recipe>

project_dir := justfile_directory()
python := project_dir / "venv/bin/python"
data_dir := project_dir / "data"
ckpt_dir := project_dir / "checkpoints/edm/supervised_R=1"

# ─── Setup ────────────────────────────────────────────────────

# Create venv and install dependencies
setup:
    python3 -m venv venv
    "{{python}}" -m pip install --upgrade pip
    "{{python}}" -m pip install -r requirements.txt

# ─── Data ─────────────────────────────────────────────────────

# Download knee singlecoil test data (requires download_data.sh with valid URLs)
download-knee:
    bash scripts/download_data.sh

# Download brain multicoil data (provide presigned URL + output filename)
# Usage: just download-brain "https://..." brain_multicoil_val_batch_1.tar.xz
download-brain url filename:
    #!/usr/bin/env bash
    set -e
    mkdir -p "{{data_dir}}"
    cd "{{data_dir}}"
    echo "Downloading {{filename}}..."
    curl -C - "{{url}}" --output "{{filename}}"
    echo "Extracting (this may take a while)..."
    tar xf "{{filename}}"
    echo "Done! Data extracted to {{data_dir}}/"

# Prepare brain data: select fully-sampled multicoil volumes for our pipeline
prep-brain input_dir="multicoil_val" num_vols="10":
    "{{python}}" scripts/prep_brain_data.py \
        --input_dir "{{data_dir}}/{{input_dir}}" \
        --output_dir "{{data_dir}}/brain_val" \
        --num_volumes {{num_vols}} \
        --target_resolution 384 320

# ─── Experiments ──────────────────────────────────────────────

# Oracle experiment on knee data
oracle-knee acc="4" n="5":
    "{{python}}" scripts/reconstruct.py \
        --mode oracle --data_path "{{data_dir}}/singlecoil_test" \
        --num_slices {{n}} --acceleration {{acc}} \
        --output_dir outputs/oracle_knee_R{{acc}}

# Oracle experiment on brain data
oracle-brain acc="4" n="5":
    "{{python}}" scripts/reconstruct.py \
        --mode oracle --data_path "{{data_dir}}/brain_val" \
        --num_slices {{n}} --acceleration {{acc}} \
        --output_dir outputs/oracle_brain_R{{acc}}

# EDM experiment on knee data
edm-knee acc="4" n="5" steps="100":
    "{{python}}" scripts/reconstruct.py \
        --mode edm --schedule edm --num_steps {{steps}} \
        --checkpoint_dir "{{ckpt_dir}}" \
        --data_path "{{data_dir}}/singlecoil_test" \
        --target_resolution 384 320 \
        --num_slices {{n}} --acceleration {{acc}} \
        --sigma_max 10.0 --device cpu \
        --output_dir outputs/edm_knee_R{{acc}}

# EDM experiment on brain data (domain-matched!)
edm-brain acc="4" n="5" steps="100":
    "{{python}}" scripts/reconstruct.py \
        --mode edm --schedule edm --num_steps {{steps}} \
        --checkpoint_dir "{{ckpt_dir}}" \
        --data_path "{{data_dir}}/brain_val" \
        --target_resolution 384 320 \
        --num_slices {{n}} --acceleration {{acc}} \
        --sigma_max 10.0 --device cpu \
        --output_dir outputs/edm_brain_R{{acc}}

# ─── Utilities ────────────────────────────────────────────────

# List available data directories
list-data:
    @ls -1d "{{data_dir}}"/*/ 2>/dev/null || echo "No data directories found."

# Show results summary from an output directory
results dir:
    @cat "{{project_dir}}/outputs/{{dir}}/results.json" | "{{python}}" -m json.tool

# Clean up: remove large archives and extracted raw data (keeps brain_val/)
clean-archives:
    rm -f "{{data_dir}}"/*.tar.xz
    rm -rf "{{data_dir}}/multicoil_test" "{{data_dir}}/multicoil_val"
