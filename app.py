from flask import Flask, render_template, request
import predict

app = Flask(__name__)


@app.route('/')
@app.route('/index.html')
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def predic():
    if request.method == "POST":
        uploaded_file = request.files['data']
        uploaded_file.save(f"static/{uploaded_file.filename}")

    s = predict.price(uploaded_file.filename)
    return render_template('prediction.html', hell=s, k=uploaded_file.filename)


if __name__ == "__main__":
    app.run(debug=True)
