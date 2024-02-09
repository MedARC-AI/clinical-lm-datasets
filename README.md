# clinical-lm-datasets

## Setup

`pip install -e .`
`python -m spacy download en_core_web_sm`

## Downloading guidelines

On local machine that has Chrome access:

- Install [Java 17](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html)
- Install [Node](https://nodejs.org/en/download/)

Commands shown for Mayo:

```
export PATH_TO_SCRAPERS='{{ INSERT DESIRED OUTPUT DIRECTORY }}
export PATH_TO_RAW=$PATH_TO_SCRAPERS"/raw"  # Raw scraped guidelines directory
export PATH_TO_CLEAN=$PATH_TO_SCRAPERS"/clean" # Clean guidelines directory
cd guidelines/scrapers/mayo
npm install --silent
tsc
node js/index.js
```

<!-- `cd guidelines && python scrapers/scrapers.py --path $PATH_TO_SCRAPERS --sources mayo` -->
