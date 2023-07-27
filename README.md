# First Principles of Computer Vision Lecture Slides

## Description
- Uploaded some sample slides of Youtube Lectures by [Prof. Shree K. Nayar](https://fpcv.cs.columbia.edu/) in this repo.
- The code to extract the slides from videos is also present in this repo and it is mainly inspired from [this repository](https://github.com/kaushikj/video2pdf)
- The code uses **OpenCV's Background Subtraction** algorighm to detect the change in the frame.
- I have modified to work it with [Prof. Shree K. Nayar's Youtube lectures](https://www.youtube.com/@firstprinciplesofcomputerv3258).
- This project converts a video presentation into a deck of pdf slides by capturing screenshots of unique frames.
Note: I have compressed the pdfs slides at the end using some online tool.

---
## Steps to run the code
`python video2pdfslides.py <video_path>`

### Run for multiple .mp4 files in a directory
`python video2pdfslides_for_multiple_vids.py <directory_path>`

## Future work
- will use Tesseract for OCR in the slides.
