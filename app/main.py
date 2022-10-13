from config import app
from flask import redirect, render_template, request, url_for
from ml_utils.mcq_utils import generate_mcqs
from ml_utils.tf_utils import generate_tf_questions
from termcolor import colored

print(colored("ALL SET ✅\n", "green"))

from config import db
from models.model import Question, Task


def save_questions_in_db(task, questions):
    for i, question in enumerate(questions):
        question, answer, options = question
        question = question.replace("<pad>", "")
        question = question.replace("</s>", "")
        question_object = Question(task_id=task.id,
                                   text=question,
                                   answer=answer,
                                   options="|".join(options))
        db.session.add(question_object)
        db.session.commit()


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        context = {}
        return render_template('index.html', **context)
    else:
        question_type = request.values.get('question_type')
        text = request.values.get('text')

        # Create a new task/job
        task = Task(title=question_type, text=text)
        db.session.add(task)
        db.session.commit()

        if question_type == 'tf':
            questions = generate_tf_questions(text)
            save_questions_in_db(task, questions)

        elif question_type == "mcqs":
            questions = generate_mcqs(text)
            save_questions_in_db(task, questions)

        return redirect(url_for('questions') + "?task_id=" + str(task.id))


@app.route('/tasks', methods=['GET'])
def tasks():
    tasks = Task.query.order_by(-Task.id).all()
    return render_template('tasks.html', tasks=tasks)


@app.route('/tasks/questions', methods=['GET'])
def questions():
    task_id = request.args.get('task_id') or 1
    task = Task.query.filter_by(id=task_id).first()
    questions = Question.query.filter_by(
        task_id=task.id).order_by(-Question.id).all()
    return render_template('questions.html', questions=questions)


if __name__ == '__main__':
    app.run(debug=True)
