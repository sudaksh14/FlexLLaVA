#!/bin/bash
#SBATCH --job-name=surfdrive-upload
#SBATCH --partition=fatq
#SBATCH --nodelist=node201
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/var/scratch/skalra/flexllava/logs/surfdrive-upload-%j.out
# Resumable upload of all checkpoints to a SURFdrive public share via WebDAV.
# Usage: sbatch scripts/upload_surfdrive.sh [TOKEN] [SUBDIR]
#   SUBDIR (e.g. baseline-tinyllama-576tok-finetune) restricts the upload to one checkpoint folder and skips its
#   checkpoint-N/ subdirs (DeepSpeed optimizer states); without it everything under checkpoints/ is uploaded.
TOKEN=${1:-w3BNexeJk9WxwCC}
SUBDIR=${2:-}
SRC=/var/scratch/skalra/flexllava/checkpoints
DAV=https://surfdrive.surf.nl/public.php/webdav
export TOKEN DAV SRC

remote_size() { # prints size or -1
  curl -s -u "$TOKEN:" -X PROPFIND -H "Depth: 0" "$DAV/$1" | grep -o '<d:getcontentlength>[0-9]*' | grep -o '[0-9]*$' || echo -1
}
export -f remote_size
upload() {
  f="$1"; rel="${f#$SRC/}"; sz=$(stat -c%s "$f")
  [ "$(remote_size "$rel")" = "$sz" ] && { echo "SKIP $rel"; return 0; }
  for i in 1 2 3; do
    code=$(curl -s -o /dev/null -w "%{http_code}" -u "$TOKEN:" -T "$f" "$DAV/$(printf %s "$rel" | sed 's/ /%20/g')")
    [[ $code == 20* ]] && [ "$(remote_size "$rel")" = "$sz" ] && { echo "OK $rel ($sz)"; return 0; }
    echo "RETRY $i $rel code=$code"; sleep 5
  done
  echo "FAIL $rel"; return 1
}
export -f upload

cd "$SRC"
SCOPE=${SUBDIR:-.}
# create all directories (MKCOL is idempotent: 405 if exists)
find "$SCOPE" -type d -not -path '*/checkpoint-*' | sed 's|^\./||' | grep -v '^\.$' | sort | while read -r d; do
  curl -s -o /dev/null -u "$TOKEN:" -X MKCOL "$DAV/$d"
done
find "$SRC/$SUBDIR" -type f -not -path '*/checkpoint-*' -print0 | xargs -0 -P 8 -I{} bash -c 'upload "{}"' | tee /var/scratch/skalra/flexllava/logs/surfdrive-upload-files.log
grep -c '^FAIL' /var/scratch/skalra/flexllava/logs/surfdrive-upload-files.log
echo DONE
