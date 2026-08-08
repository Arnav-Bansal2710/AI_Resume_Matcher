from flask import Flask, render_template, request

from model.preprocess import (
    extract_text_from_pdf,
    clean_text
)

from model.similarity import (
    calculate_final_score,
    get_missing_keywords
)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():

    score = None
    missing_keywords = []

    if request.method == "POST":

        file = request.files.get("resume")
        job_desc = request.form.get("job_desc", "")

        if file and job_desc:

            resume_text = extract_text_from_pdf(file)

            resume_text = clean_text(resume_text)
            job_desc = clean_text(job_desc)

            score = calculate_final_score(
                resume_text,
                job_desc
            )

            missing_keywords = get_missing_keywords(
                resume_text,
                job_desc
            )

    return render_template(
        "index.html",
        score=score,
        missing_keywords=missing_keywords
    )


if __name__ == "__main__":
    app.run(debug=True)