#!/usr/bin/env bash
# Download the MVLAD-AD weights that this fork only holds as Git LFS pointers.
#
# The fork's LFS store returns 404, but the official repository (lan-qing/MVLAD-AD) serves
# the same objects. Each pointer file records the object's sha256 and size; this script
# downloads the object, checks both, and only then replaces the pointer.
#
#   bash scripts/download_mvlad_weights.sh            # all weights (~800 MB)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="https://media.githubusercontent.com/media/lan-qing/MVLAD-AD/main"
FILES=(
    tokenizer/tokenizer_model.pth
    weights/planning/adapter_model.safetensors
    weights/planning/non_lora_trainables.bin
    weights/nu_x/adapter_model.safetensors
    weights/nu_x/non_lora_trainables.bin
)

for rel in "${FILES[@]}"; do
    dest="$REPO_ROOT/$rel"
    if ! head -c 100 "$dest" | grep -q "git-lfs.github.com/spec"; then
        echo "skip  $rel (already a real file)"
        continue
    fi
    oid=$(sed -n 's/^oid sha256:\([0-9a-f]*\)$/\1/p' "$dest")
    size=$(sed -n 's/^size \([0-9]*\)$/\1/p' "$dest")
    tmp="$dest.download"
    echo "fetch $rel ($size bytes)"
    curl -fL --retry 4 --retry-delay 2 -o "$tmp" "$BASE_URL/$rel"
    got_size=$(stat -c %s "$tmp")
    got_oid=$(sha256sum "$tmp" | cut -d' ' -f1)
    if [ "$got_size" != "$size" ] || [ "$got_oid" != "$oid" ]; then
        echo "ERROR: $rel does not match its LFS pointer (size $got_size vs $size, sha256 $got_oid vs $oid)" >&2
        exit 1
    fi
    mv "$tmp" "$dest"
    echo "ok    $rel"
done
echo "All weights verified against their LFS pointers."
echo "Note: git now sees these files as modified; don't commit them."
