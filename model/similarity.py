import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Skills that the matcher knows how to detect
skills_vocab = [
    # Programming Languages
    "python",
    "java",
    "javascript",
    "typescript",
    "c++",
    "c",

    # Frontend
    "html",
    "css",
    "react",
    "react.js",
    "angular",
    "vue",
    "tailwind",
    "tailwind css",
    "bootstrap",

    # Backend
    "node.js",
    "node",
    "express",
    "express.js",
    "flask",
    "django",
    "spring",
    "rest api",
    "rest apis",

    # Databases
    "mysql",
    "postgresql",
    "mongodb",
    "sql",
    "redis",

    # AI / Machine Learning
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "nlp",
    "scikit-learn",
    "scikit learn",
    "tensorflow",
    "pytorch",
    "pandas",
    "numpy",
    "transformers",
    "bert",

    # DSA
    "data structures",
    "algorithms",
    "dynamic programming",
    "graphs",
    "trees",
    "recursion",

    # Tools / DevOps
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "vercel",
    "postman",

    # Backend / Security concepts
    "jwt",
    "api",
    "rest",
    "authentication",
    "authorization"
]


def extract_skills(text):
    """
    Extract known skills from text.
    """

    found = set()

    text = text.lower()

    for skill in skills_vocab:

        pattern = r'\b' + re.escape(skill.lower()) + r'\b'

        if re.search(pattern, text):
            found.add(skill)

    return found


def calculate_similarity(resume_text, job_desc):
    """
    Calculate TF-IDF cosine similarity between
    resume and job description.
    """

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    documents = [
        resume_text,
        job_desc
    ]

    tfidf_matrix = tfidf.fit_transform(documents)

    score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    return round(score[0][0] * 100, 2)


def calculate_skill_score(resume_text, job_desc):
    """
    Calculate the percentage of required job skills
    that are present in the resume.
    """

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_desc)

    # If no recognized skills exist in the JD,
    # we cannot calculate a skill score.
    if not jd_skills:
        return 0

    matched_skills = resume_skills.intersection(jd_skills)

    score = (
        len(matched_skills) /
        len(jd_skills)
    ) * 100

    return round(score, 2)


def calculate_final_score(resume_text, job_desc):
    """
    Combine text similarity and skill matching.

    40% -> TF-IDF text similarity
    60% -> Required skill matching
    """

    text_score = calculate_similarity(
        resume_text,
        job_desc
    )

    skill_score = calculate_skill_score(
        resume_text,
        job_desc
    )

    final_score = (
        0.4 * text_score +
        0.6 * skill_score
    )

    return round(final_score, 2)


def get_missing_keywords(resume_text, job_desc):
    """
    Find skills required by the job description
    that are missing from the resume.
    """

    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(job_desc)

    missing = jd_skills - resume_skills

    return sorted(list(missing))