
import difflib

def save_side_by_side_diff_to_html(old_content, new_content, output_file):
    d = difflib.HtmlDiff()
    diff_html = d.make_file(old_content.splitlines(), new_content.splitlines(), context=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(diff_html)

if __name__ == "__main__":
    # 比較するファイルの内容を取得
    file1_name = r'../Ais4ToCurForKalmanTime-old.py'
    file2_name = r'../Kalman.py'
    with open(file1_name, "r", encoding='utf-8') as file1, open(file2_name, "r", encoding='utf-8') as file2:
        content1 = file1.read()
        content2 = file2.read()

    # 差分をHTMLファイルに保存（変更前を左、変更後を右に表示）
    save_side_by_side_diff_to_html(content1, content2, "side_by_side_diff_output.html")