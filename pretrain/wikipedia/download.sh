TO_DIR=/weka/home-griffin/clinical_pile/wikipedia/

aws s3 cp s3://pile-everything-west/dolma/v1/wiki-en-simple/en_simple_wiki-0000.json.gz $TO_DIR
aws s3 cp s3://pile-everything-west/dolma/v1/wiki-en-simple/en_simple_wiki-0001.json.gz $TO_DIR
