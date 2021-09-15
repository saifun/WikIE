# WikIE
A repo for the final project in Hebrew NLP - extracting information from biographies to enrich Wikipedia

## Running the code
* Before you run the code, download [this](https://drive.google.com/file/d/1MeP9ea7uq22vuMEUTUR4JS9ZB-5-Xqnd/view?usp=sharing) zip file and extract it inside the folder `model_data`.
* Install the package locally - `pip install -e src`

When running the code, make sure you are inside `src` directory.

Code example for using inline text:
```
import wikie
ie = wikie.IE()
text = 'אלברט איינשטיין נולד בגרמניה וגר בשוויץ'  # Type hebrew biography text to extract information from.
ie.extract_text_information(text)
```

Code example for reading text from a file:
```
import wikie
ie = wikie.IE()
ie.extract_information_from_file(<file_path>)
```

