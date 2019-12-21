pip install pretrainedmodels

export PYTHONPATH=${PYTHONPATH}:/kaggle/working && python setup.py develop  --install-dir /kaggle/working

download_from_gdrive() {
    file_id=$1
    file_name=$2

    # first stage to get the warning html
    curl -c /tmp/cookies \
    "https://drive.google.com/uc?export=download&id=$file_id" > \
    /tmp/intermezzo.html

    # second stage to extract the download link from html above
    download_link=$(cat /tmp/intermezzo.html | \
    grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | \
    sed 's/\&amp;/\&/g')
    curl -L -b /tmp/cookies \
    "https://drive.google.com$download_link" > $file_name
}

mkdir -q model_seresnext101
download_from_gdrive '1CCT2MmeN_mNZkIjzkfL9DKZSjESYIs4M' 'model_seresnext101/best-model.pt'
download_from_gdrive '14w2MYhq-bb03Rz4Z572j9dwkfi0NgEtc' 'model_seresnext101/model.pt'