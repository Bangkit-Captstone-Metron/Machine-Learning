## Metron Backend

### Features:
* OCR
* Image Forgery Detection (Coming soon)

### Installation

1. Install Docker
1. Build the Docker image:
```docker image build --tag metron_backend .```
1. Run the App:
```docker container run --publish 5000:5000 metron_backend```

Note: Stop the app using Ctrl-C.

## Usage

* OCR

```
Path: /v1/ocr
Method: POST
Body: JSON
{
    "image_url": "http://...."
}
Result:
{
    "data": "..."
}
```

## TODO
1. Add preprocessing to improve OCR Accuracy,
1. Add image forgery detection