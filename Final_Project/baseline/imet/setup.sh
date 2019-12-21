git clone https://github.com/lincw6666/Computer_Vision_Based_on_DeepLearning.git --branch final_project --single-branch --depth 1
mv Computer_Vision_Based_on_DeepLearning/Final_Project/baseline/* ./
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

download_from_gdrive '1PAiDMDPNcj-McEI62CqK7PTCrkYplaU4' 'model_seresnext101'