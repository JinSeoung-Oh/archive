import subprocess
import tempfile
import os
import re
import json

def check_latex_syntax(latex_code):
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_file = os.path.join(tmpdir, 'temp.tex')

        # LaTeX 프리앰블 작성
        latex_preamble = r'''\documentclass{article}
\usepackage{xeCJK}
\setCJKmainfont{Apple SD Gothic Neo}
\begin{document}
'''
        latex_end = r'\end{document}'

        # LaTeX 코드를 파일에 작성
        with open(tex_file, 'w', encoding='utf-8') as f:
            f.write(latex_preamble)
            f.write(latex_code)
            f.write('\n' + latex_end)

        # xelatex 실행

        try:
            subprocess.run(
                ["/Library/TeX/texbin/xelatex", '-interaction=nonstopmode', tex_file],
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=True
            )
            # 컴파일 성공 시 문법 오류 없음
            return True, None
        except subprocess.CalledProcessError as e:
            # 컴파일 실패 시 오류 메시지 추출
            output = e.stdout.decode('utf-8', errors='ignore')
            error_message, error_snippet = parse_latex_log(output)
            char_pos = find_error_position(latex_code, error_snippet)
            return False, {
                'error_message': error_message,
                'char_position': char_pos,
                'error_snippet': error_snippet
            }

def parse_latex_log(log_content):
    # 오류 메시지와 오류 발생한 부분 추출
    lines = log_content.splitlines()
    error_message = []
    error_snippet = ''
    capture = False
    for line in lines:
        if line.startswith('!'):
            capture = True
            error_message.append(line)
        elif capture:
            if line.strip() == '':
                break  # 빈 줄이면 오류 메시지 끝
            else:
                error_message.append(line)
                # 오류가 발생한 줄에서 코드 부분 추출
                if 'l.' in line:
                    match = re.search(r'l\.\d+\s*(.*)', line)
                    if match:
                        error_snippet = match.group(1).strip()
    return '\n'.join(error_message), error_snippet

def find_error_position(latex_code, error_snippet):
    if not error_snippet:
        return -1  # 오류 부분을 찾을 수 없음
    # 오류 부분에서 LaTeX이 추가한 내용 제거
    error_snippet_clean = error_snippet.replace('<to be read again>', '').strip()
    # 원본 코드에서 오류 부분의 위치 찾기
    index = latex_code.find(error_snippet_clean)
    if index != -1:
        return index
    else:
        # 공백 제거 후 검색
        index = latex_code.replace(' ', '').find(error_snippet_clean.replace(' ', ''))
        if index != -1:
            return index
    return -1  # 찾을 수 없음

def extract_latex_with_xelatex(text):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.tex')
        output_file = os.path.join(tmpdir, 'math.txt')
        
        # LaTeX 문서 생성
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(r'''\documentclass{article}
\usepackage{verbatim}
\usepackage{fancyvrb}
\newwrite\mathfile
\immediate\openout\mathfile=math.txt
\makeatletter
\newcommand{\extractmath}[1]{%
  \immediate\write\mathfile{MATH_START}%
  \immediate\write\mathfile{\unexpanded{#1}}%
  \immediate\write\mathfile{MATH_END}%
}
\catcode`$=\active
\def${\futurelet\next\extract@math}
\def\extract@math{%
  \ifx\next$%
    \expandafter\extract@display@math
  \else
    \expandafter\extract@inline@math
  \fi
}
\def\extract@inline@math#1${%
  \extractmath{#1}%
}
\def\extract@display@math$#1$${%
  \extractmath{#1}%
}
\makeatother
\begin{document}
''')
            f.write(text)
            f.write(r'\end{document}')
        
        # xelatex 실행
        subprocess.run(
            ["/Library/TeX/texbin/xelatex", '-interaction=nonstopmode', input_file],
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 수식 파일 읽기
        latex_parts = []
        current_math = []
        in_math = False
        
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == 'MATH_START':
                    in_math = True
                elif line.strip() == 'MATH_END':
                    if current_math:
                        latex_parts.append(''.join(current_math))
                        current_math = []
                    in_math = False
                elif in_math:
                    current_math.append(line.rstrip('\n'))
        
        return latex_parts


def find_all_folder_paths(directory):
    folder_list = []
    items = os.listdir(directory)

    for item in items:
        item_path = os.path.join(directory, item)
        folder_list.append(item_path)
    
    return folder_list

def determin(is_vaild, error_info, latex_code):
    check_result = {}
    if is_vaild:
        check_result['vaild'] = "T"
        check_result['error'] = "None"
    else:
        check_result['vaild'] = "N"
        if error_info['char_position'] != -1:
            error_part = latex_code[error_info['char_position']:error_info['char_position']+len(error_info['error_snippet'])]
            check_result['error'] = error_part
        else:
            check_result['error'] = "LaTex 문법에 어긋나는 부분이 있으나 정확한 위치를 파악하지 못 하였습니다."
    return check_result

def append_to_jsonl(file_path, data):
    with open(file_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False,indent=4)


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line.strip())
            data.append(json_object)
    return data

def extract_latex_with_xelatex(text):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'input.tex')
        output_file = os.path.join(tmpdir, 'math.txt')
        
        # LaTeX 문서 생성
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(r'''\documentclass{article}
\usepackage{verbatim}
\newwrite\mathfile
\immediate\openout\mathfile=math.txt
\makeatletter
\newcommand{\extractmath}[1]{%
  \immediate\write\mathfile{MATH_START}%
  {\let\do\@makeother\dospecials\immediate\write\mathfile{#1}}%
  \immediate\write\mathfile{MATH_END}%
}
\catcode`$=\active
\def${\futurelet\next\extract@math}
\def\extract@math{%
  \ifx\next$%
    \expandafter\extract@display@math
  \else
    \expandafter\extract@inline@math
  \fi
}
\def\extract@inline@math#1${%
  \extractmath{#1}%
  #1$%
}
\def\extract@display@math$#1$${%
  \extractmath{#1}%
  #1$$%
}
\makeatother
\begin{document}
''')
            f.write(text)
            f.write(r'\end{document}')
        
        # xelatex 실행
        subprocess.run(
            ["/Library/TeX/texbin/xelatex", '-interaction=nonstopmode', input_file],
            cwd=tmpdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # 수식 파일 읽기
        latex_parts = []
        current_math = []
        in_math = False
        
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == 'MATH_START':
                    in_math = True
                elif line.strip() == 'MATH_END':
                    if current_math:
                        latex_parts.append(''.join(current_math))
                        current_math = []
                    in_math = False
                elif in_math:
                    current_math.append(line.rstrip('\n'))
        
        return latex_parts

def compare(str_1, str_2):
    differ = []
    ori_ = extract_latex_with_xelatex(str_1)
    step2_ = extract_latex_with_xelatex(str_2)
    if ori_:
        ori = ori_[0]
    if not ori_:
        ori = ['DO NOT CONTAIN LATEX']
        
    if step2_:
        step2 = step2_[0]
    if not step2_:
        step2 = ['DO NOT CONTAIN LATEX']
        
    common_length = min(len(ori), len(step2))
    
    for i in range(common_length):
        if ori[i] != step2[i]:
            differ.append(i)

    if len(ori) != len(step2):
        for i in range(common_length, max(len(ori), len(step2))):
            differ.append(i)
    
    if differ:
        ran = [min(differ), max(differ)]
    else:
        ran = None
    return ran

save_re_result = './_수학_체크/'
data = read_jsonl('./PROJ-12367_24733_Task4_normal_resultData_2024-10-02_17-21-12.jsonl')


for i in range(len(data)):
    print(i)
    data_ = data[i]
    index = data_['index']
    print(index)
    ori_q = data_['origin_question']
    ori_s = data_['origin_solution']
    ori_a = data_['origin_answer']
    check_q = data_['step2_question']
    check_s = data_['step2_solution']
    check_a = data_['step2_answer']

    compare_q = compare(ori_q, check_q)
    compare_s = compare(ori_s, check_s)
    compare_a = compare(ori_a, check_a)
    
    oq_is_vaild, oq_error_info = check_latex_syntax(ori_q)
    os_is_vaild, os_error_info = check_latex_syntax(ori_s)
    oa_is_vaild, oa_error_info = check_latex_syntax(ori_a)
    q_is_valid, q_error_info = check_latex_syntax(check_q)
    s_is_vaild, s_error_info = check_latex_syntax(check_s)
    a_is_vaild, a_error_info = check_latex_syntax(check_a)

    oq_determin = determin(oq_is_vaild, oq_error_info, ori_q)
    os_determin = determin(os_is_vaild, os_error_info, ori_s)
    oa_determin = determin(oa_is_vaild, oa_error_info, ori_a)
    q_determin = determin(q_is_valid, q_error_info, check_q)
    s_determin = determin(s_is_vaild, s_error_info, check_s)
    a_determin = determin(a_is_vaild, a_error_info, check_a)

    o_determin_result = []
    o_determin_result.append(oq_determin)
    o_determin_result.append(os_determin)
    o_determin_result.append(oa_determin)
    determin_result = []
    determin_result.append(q_determin)
    determin_result.append(s_determin)
    determin_result.append(a_determin)

    if not compare_q:
        compare_result = {"differ" : "N","position":"None"}
    if compare_q:
        compare_result = {"differ" : "T", "position":compare_q}
    data_['compare_q'] = compare_result

    if compare_s:
        compare_result = {"differ":"T", "position":compare_s}
    if not compare_s:
        compare_result = {"differ":"N", "position":"None"}
    data_['compare_s'] = compare_result
    
    if compare_a:
        compare_result = {"differ":"T", "position":compare_a}
    if not compare_a:
        compare_result = {"differ":"N", "position":"None"}
    data_['compare_a'] = compare_result
    
    data_['origin_check_result'] = o_determin_result
    data_['step2_check_result'] = determin_result

    print(data_)
    save_path = save_re_result + str(index) + '.json'
    append_to_jsonl(save_path, data_)
