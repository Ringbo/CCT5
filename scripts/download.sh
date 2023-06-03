#!/bin/bash

#!/bin/bash

if [ -e "CCT5.tar.bz2.aa" ] || [ -e "CCT5.tar.bz2.ab" ] || [ -e "CCT5.tar.bz2.ac" ] || [ -e "CCT5.tar.bz2.ad" ] || [ -e "CCT5.tar.bz2.ae" ] || [ -e "CCT5.tar.bz2.af" ] || [ -e "CCT5.tar.bz2.ag" ] || [ -e "CCT5.tar.bz2.ah" ] || [ -e "CCT5.tar.bz2.ai" ] || [ -e "CCT5.tar.bz2.aj" ]
then
    read -p "cache file existed, remove it?(y/n) " choice
    if [ "$choice" == "y" ]; then
        rm CCT5.tar.bz2
        echo "Cache file has been deleted."
    else
        echo "Keep cache file."
    fi
else
    echo "Downloading dataset"
fi

wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.aa
wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.ab
wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.ac
wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.ad
wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.ae
wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.af
wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.ag
wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.ah
wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.ai
wget https://www.zenodo.org/record/7964370/files/CCT5.tar.bz2.aj
cat CCT5.tar.bz2.* > CCT5.tar.bz2
md5=$(md5sum CCT5.tar.bz2 | awk '{print $1}')
if [ "$md5" == "13fca7020b532cddf065eb726973e0cb" ]; then
    echo "MD5 check successful."
else
    echo "MD5 check failed. Please delete the downloaded file and re-run the script."
    exit 1
fi
tar -axf CCT5.tar.bz2
wget https://www.zenodo.org/record/7998509/files/CCT5-v1-patch.tar.bz2
tar -axf CCT5-v1-patch.tar.bz2