from config import db


class Task(db.Model):
    __tablename__ = 'task'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    text = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return '<Job %r>' % self.title

    def __str__(self):
        return self.title


class Question(db.Model):
    __tablename__ = 'question'
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('task.id'))
    task = db.relationship("Task",
                           backref=db.backref("request", uselist=False))
    text = db.Column(db.String(100), nullable=True)
    answer = db.Column(db.String(100), nullable=True)
    options = db.Column(db.String(100), nullable=True)

    def __repr__(self):
        return '<Question %r>' % self.text

    def __str__(self):
        return self.text
    
    def get_option_list(self):
        return self.options.split("|")


db.create_all()
db.session.commit()
