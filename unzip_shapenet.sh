
#!/bin/bash

# Change this to the path where your zip files are located by argument
ZIP_PATH=$1

# Loop through all zip files in the directory and unzip them
for file in "$ZIP_PATH"/*.zip
do
    unzip "$file" -d "${file%.*}"
done
