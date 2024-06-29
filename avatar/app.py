from flask import Flask, send_file, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video/<filename>')
def stream_video(filename):
    path = f'video/{filename}'
    return send_file(path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)