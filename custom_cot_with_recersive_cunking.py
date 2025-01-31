import re
import requests
import time
import json
from json_repair import repair_json
import anthropic
import getpass
import os
from tenacity import retry, stop_after_attempt, wait_exponential

def find_all_folder_paths(directory):
    folder_list = []
    items = os.listdir(directory)

    for item in items:
        item_path = os.path.join(directory, item)
        folder_list.append(item_path)
    
    return folder_list

client_ = anthropic.Anthropic(api_key="sk-ant-...")
 
def get_response_from_claude(prompt_templet, context):
    result_text = ""
    
    # Claude에 메시지 생성 요청을 보냅니다.
    response = client_.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        temperature=0.0,
        system= prompt_templet,
        messages=[{"role": "user", "content": f'주어진 지침을 철저하게 따라 {context}로부터 QA 데이터를 만들어주세요.'}]
    )
    
    # 응답 객체에서 텍스트 내용만 추출합니다.
    if not response.content or not isinstance(response.content, list):
        result_text = "No response or unexpected response format."
    else:
        response_texts = [block.text for block in response.content if hasattr(block, 'text')]
        result_text = " ".join(response_texts)
 
    return result_text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_response_with_retry(prompt_templet,context):
    try:
        return get_response_from_claude(prompt_templet,context)
    except Exception as e:
        print(f"Error occurred: {str(e)}. Retrying...")
        time.sleep(random.uniform(1, 3))  # Random delay before retry
        raise  # Re-raise the exception to trigger a retry

def nested_list_to_string(nested_list):
    # 각 내부 리스트를 문자열로 변환
    string_lists = [''.join(inner_list) for inner_list in nested_list]
    
    # 변환된 문자열들을 하나로 연결
    result = ''.join(string_lists)
    
    return result

custom_cot_gen_prompt = """
당신은 고성능 질의응답(QA) 시스템입니다. 주어진 문서를 기반으로 질문을 생성하고 답변해야 합니다. 다음 지침을 엄격히 따라주세요:

1. 정보 소스
   반드시 제공된 문서의 정보만을 사용하세요.
   외부 지식이나 개인적 의견을 포함하지 마세요.

2. 질문 생성
   주어진 텍스트를 바탕으로 다양한 시나리오의 질문을 생성하세요.
   다음 기준을 따르세요:
   a) 하나의 문서에 최대 3개의 Chain of Thought (CoT) 데이터를 허용합니다.
   b) 3개의 질문은 다양성을 보장해야 하며, 문서의 서로 다른 부분에 대한 것이어야 합니다.
   c) 단순한 답변이 아닌, 단계별 추론이 필요한 복합추론용 질문을 만드세요.
   d) 문서의 최소 두 곳의 정보를 참조하여 답변할 수 있는 질문을 만드세요.
   질문 형식: "[질문 내용]"

3. 답변 구조 (CoT 추론 순서)
   모든 답변은 다음 CoT 추론 순서를 엄격히 따라야 합니다:
   a) 질의 재확인: 질문의 핵심 내용을 간략히 재진술합니다.
   b) 답변이 포함된 문단 정보 확인: 
      답변에 필요한 정보가 문서의 어느 부분에서 확인 가능한지 명시합니다. 되도록이면 자세하게 명시해주세요.
      예시) 한국의 온실가스 총배출량이 1990년 이후 감소한 연도는 <국내 배출 추이 분석 및 부문별 배출량 분석>에서 확인할 수 있으며, 2030년 감축 목표는 <연구배경>에서 파악할 수 있습니다
   c) 질의에 대한 답변 제시:
      질문에 대한 직접적인 답변을 제시합니다.
      필요한 경우, 답변을 구조화하여 제시합니다(예: 불릿 포인트 사용).
      계산이 필요한 경우, 단계별 계산 과정을 상세히 기술하세요.
      답변은 비교적 자세하게 해주세요.
   d) (선택사항) 최종 답변 제공: 필요한 경우, 답변에 대한 간략한 요약이나 결론을 제시합니다.

4. 답변 작성 시 주의사항
   객관성을 유지하고, 문서에 명시된 사실만을 바탕으로 답변하세요.
   문서에서 직접 도출할 수 있는 정보만 사용하세요.
   모든 답변은 제공된 문서의 내용에 기반해야 합니다.
   계산이 필요한 경우, 반드시 상세한 단계별 과정을 보여주세요.

이 지침을 따라 주어진 문서를 바탕으로 다양한 질문을 생성하고, 이에 대해 CoT 추론 순서에 따라 구조화되고 상세한 답변을 제공해 주세요.
QA 외에 다른 쓸데 없는 정보는 리턴하지 말아주세요.

다음과 같은 json object로 리턴해주세요.
[{"질문" : "질문 내용", "답변" : "답변 구조를 만족하는 답변"}, ..., {"질문" : "질문 내용", "답변" : "답변 구조를 만족하는 답변"}]

"""

def remove_newlines_in_cells(markdown_text):
    # 각 셀 내에 존재하는 줄바꿈(\n)을 공백으로 변환
    return re.sub(r'(?<=\|)([^|]+)\n([^|]+)(?=\|)', r'\1 \2', markdown_text)

def chunk_text_optimized(text, min_chunk_size=4000, max_chunk_size=4700, overlap_ratio=0.2):
    lines = text.split('\n')
    total_lines = len(lines)
    chunks = []
    i = 0

    def is_table_row(line):
        return line.strip().startswith("|") and line.strip().endswith("|")

    def get_text_length(chunk_lines):
        return sum(len(line) + 1 for line in chunk_lines)  # +1 for '\n'

    def find_table_end(start_idx):
        table_lines = []
        idx = start_idx
        while idx < total_lines:
            line = lines[idx]
            if is_table_row(line) or not line.strip():
                table_lines.append(line)
                idx += 1
            else:
                break
        return idx, table_lines

    next_chunk_start = []

    while i < total_lines:
        current_chunk = next_chunk_start
        next_chunk_start = []
        current_length = get_text_length(current_chunk)
        removed_lines = []

        while i < total_lines and current_length < max_chunk_size:
            line = lines[i]
            line_length = len(line) + 1  # +1 for '\n'

            if is_table_row(line):
                # 테이블의 시작을 찾음
                table_end_idx, table_lines = find_table_end(i)
                table_length = get_text_length(table_lines)

                if current_length + table_length <= max_chunk_size or current_length == 0:
                    current_chunk.extend(table_lines)
                    current_length += table_length
                    i = table_end_idx
                else:
                    if current_length == 0:
                        # 현재 청크가 비어있다면 테이블 전체를 포함
                        current_chunk.extend(table_lines)
                        current_length += table_length
                        i = table_end_idx
                    break  # 청크가 꽉 찼으므로 종료
            else:
                if current_length + line_length <= max_chunk_size:
                    current_chunk.append(line)
                    current_length += line_length
                    i += 1
                else:
                    break  # 청크가 꽉 찼으므로 종료

        # 청크의 크기가 max_chunk_size를 초과하는 경우, 테이블이 아닌 라인을 제거하여 크기를 줄임
        while current_length > max_chunk_size:
            if current_chunk and not is_table_row(current_chunk[-1]):
                removed_line = current_chunk.pop()
                next_chunk_start.insert(0, removed_line)  # 제거된 라인은 다음 청크에 포함
                current_length -= len(removed_line) + 1
            else:
                break  # 더 이상 제거할 수 있는 라인이 없음

        # 오버랩 처리
        overlap_lines = []
        if overlap_ratio > 0:
            overlap_count = int(len(current_chunk) * overlap_ratio)
            overlap_lines = current_chunk[-overlap_count:] if overlap_count > 0 else []

            # 오버랩에 테이블이 있다면 테이블 전체를 포함
            if any(is_table_row(line) for line in overlap_lines):
                idx_overlap = len(current_chunk) - overlap_count - 1
                while idx_overlap >= 0 and is_table_row(current_chunk[idx_overlap]):
                    overlap_lines.insert(0, current_chunk[idx_overlap])
                    idx_overlap -= 1

        # 현재 청크를 추가
        chunks.append(current_chunk)

        # 다음 청크의 시작 부분에 오버랩 라인과 제거된 라인을 추가
        next_chunk_start = overlap_lines + next_chunk_start

    # 청크를 문자열로 변환
    chunk_texts = ['\n'.join(chunk) for chunk in chunks if chunk]

    # 최소 청크 크기 확인 및 조정
    adjusted_chunks = []
    idx = 0
    while idx < len(chunk_texts):
        chunk = chunk_texts[idx]
        chunk_length = len(chunk)
        if chunk_length < min_chunk_size and idx > 0:
            # 이전 청크에서 필요한 만큼 가져옴
            prev_chunk = adjusted_chunks.pop()
            prev_chunk_lines = prev_chunk.split('\n')
            current_chunk_lines = chunk.split('\n')

            while get_text_length(current_chunk_lines) < min_chunk_size and prev_chunk_lines:
                line = prev_chunk_lines.pop()
                if is_table_row(line):
                    # 테이블이 분할되지 않도록 테이블의 시작까지 가져옴
                    table_lines = [line]
                    while prev_chunk_lines and is_table_row(prev_chunk_lines[-1]):
                        table_lines.insert(0, prev_chunk_lines.pop())
                    current_chunk_lines = table_lines + current_chunk_lines
                else:
                    current_chunk_lines.insert(0, line)

            adjusted_chunks.append('\n'.join(prev_chunk_lines))
            adjusted_chunks.append('\n'.join(current_chunk_lines))
        else:
            adjusted_chunks.append(chunk)
        idx += 1

    # 마지막 청크의 크기가 min_chunk_size보다 작은 경우 처리
    if adjusted_chunks:
        last_chunk = adjusted_chunks[-1]
        if len(last_chunk) < min_chunk_size and len(adjusted_chunks) > 1:
            # 이전 청크에서 필요한 만큼 가져옴
            prev_chunk = adjusted_chunks.pop(-2)
            prev_chunk_lines = prev_chunk.split('\n')
            last_chunk_lines = last_chunk.split('\n')

            while get_text_length(last_chunk_lines) < min_chunk_size and prev_chunk_lines:
                line = prev_chunk_lines.pop()
                if is_table_row(line):
                    # 테이블이 분할되지 않도록 처리
                    table_lines = [line]
                    while prev_chunk_lines and is_table_row(prev_chunk_lines[-1]):
                        table_lines.insert(0, prev_chunk_lines.pop())
                    last_chunk_lines = table_lines + last_chunk_lines
                else:
                    last_chunk_lines.insert(0, line)

            adjusted_chunks[-2] = '\n'.join(prev_chunk_lines)
            adjusted_chunks[-1] = '\n'.join(last_chunk_lines)

    # 빈 청크 제거
    adjusted_chunks = [chunk for chunk in adjusted_chunks if chunk.strip()]

    return adjusted_chunks

def split_chunk_into_subchunks(chunk, first_chunk_size, min_chunk_size=4000, max_chunk_size=5000, overlap_ratio=0.2):
    lines = chunk.split('\n')
    total_lines = len(lines)
    sub_chunk1_lines = []
    sub_chunk2_lines = []
    current_length = 0
    i = 0  # 라인 인덱스

    def is_table_row(line):
        return line.strip().startswith('|') and line.strip().endswith('|')

    def get_text_length(lines_list):
        return sum(len(line) + 1 for line in lines_list)  # +1은 '\n'을 위한 것

    def find_table_end(start_idx):
        idx = start_idx
        while idx < total_lines and (is_table_row(lines[idx]) or not lines[idx].strip()):
            idx += 1
        return idx

    # **첫 번째 서브 청크 생성**
    while i < total_lines and current_length < first_chunk_size:
        line = lines[i]
        line_length = len(line) + 1  # '\n' 포함

        if is_table_row(line):
            # 테이블의 끝을 찾습니다.
            table_end_idx = find_table_end(i)
            table_lines = lines[i:table_end_idx]
            table_length = get_text_length(table_lines)

            if current_length + table_length <= max_chunk_size or current_length == 0:
                # 테이블을 첫 번째 서브 청크에 추가합니다.
                sub_chunk1_lines.extend(table_lines)
                current_length += table_length
                i = table_end_idx
            else:
                break  # 테이블을 추가하면 max_chunk_size를 초과하므로 종료
        else:
            if current_length + line_length <= max_chunk_size:
                sub_chunk1_lines.append(line)
                current_length += line_length
                i += 1
            else:
                break

    # 최소 청크 크기 보장
    if current_length < min_chunk_size:
        while i < total_lines and current_length < min_chunk_size:
            line = lines[i]
            line_length = len(line) + 1

            if is_table_row(line):
                table_end_idx = find_table_end(i)
                table_lines = lines[i:table_end_idx]
                table_length = get_text_length(table_lines)
                sub_chunk1_lines.extend(table_lines)
                current_length += table_length
                i = table_end_idx
            else:
                sub_chunk1_lines.append(line)
                current_length += line_length
                i += 1

    # **오버랩 처리**
    overlap_length = int(get_text_length(sub_chunk1_lines) * overlap_ratio)
    print(overlap_length)
    temp_length = 0
    idx_overlap = len(sub_chunk1_lines) - 1
    overlap_lines = []

    while idx_overlap >= 0 and temp_length < overlap_length:
        line = sub_chunk1_lines[idx_overlap]
        overlap_lines.insert(0, line)
        temp_length += len(line) + 1
        idx_overlap -= 1

    # 오버랩에 테이블이 포함되면 테이블 전체를 포함
    if any(is_table_row(line) for line in overlap_lines):
        while idx_overlap >= 0 and is_table_row(sub_chunk1_lines[idx_overlap]):
            line = sub_chunk1_lines[idx_overlap]
            overlap_lines.insert(0, line)
            idx_overlap -= 1
            
    # **두 번째 서브 청크 생성**
    sub_chunk2_lines = overlap_lines + lines[i:]

    # 두 번째 서브 청크의 크기 조정
    sub_chunk2_length = get_text_length(sub_chunk2_lines)

    if sub_chunk2_length > max_chunk_size:
        # 두 번째 서브 청크를 max_chunk_size 이하로 조정
        current_length = 0
        idx = 0
        sub_chunk2_final_lines = []
        while idx < len(sub_chunk2_lines) and current_length < max_chunk_size:
            line = sub_chunk2_lines[idx]
            line_length = len(line) + 1

            if is_table_row(line):
                # 테이블의 끝을 찾습니다.
                table_end_idx = idx + 1
                while table_end_idx < len(sub_chunk2_lines) and (is_table_row(sub_chunk2_lines[table_end_idx]) or not sub_chunk2_lines[table_end_idx].strip()):
                    table_end_idx += 1
                table_lines = sub_chunk2_lines[idx:table_end_idx]
                table_length = get_text_length(table_lines)
                if current_length + table_length <= max_chunk_size or current_length == 0:
                    sub_chunk2_final_lines.extend(table_lines)
                    current_length += table_length
                    idx = table_end_idx
                else:
                    break
            else:
                if current_length + line_length <= max_chunk_size:
                    sub_chunk2_final_lines.append(line)
                    current_length += line_length
                    idx += 1
                else:
                    break
        sub_chunk2_lines = sub_chunk2_final_lines

    # 라인들을 문자열로 변환
    sub_chunk1 = '\n'.join(sub_chunk1_lines)
    sub_chunk2 = '\n'.join(sub_chunk2_lines)

    return sub_chunk1, sub_chunk2

path = './Task_1_data'
files = find_all_folder_paths(path)
save_path = './chunking/'

for file in files:
    name = file.split('/')[-1].split('.')[-2]
    with open(file, 'r') as f:
        data = json.load(f)
    all_text = ''
    for i in range(len(data['doc_body'])):
        data_ = data['doc_body'][i]['content']
        all_text += data_ + '\n'
    all_text = remove_newlines_in_cells(all_text)
    chunk_list = chunk_text_optimized(all_text)
            
    for k in range(len(chunk_list)):
        chunk = chunk_list[k]
        if len(chunk) < 4000 or len(chunk) > 5000:
            new_chunk = chunk_list[k-1] + '\n\n' + chunk_list[k]
            print('.................................................................................................')
            print(name + '_' + str(k))
            chunk = split_chunk_into_subchunks(new_chunk, len(chunk_list[k-1]))
            with open(save_path + name + '_' + str(k) + '.json', 'w') as f:
                json.dump(chunk[1], f, ensure_ascii=False, indent=4)
            chunk_list[k] = chunk[1]
        else:
            print(len(chunk))
            with open(save_path + name + '_' + str(k) + '.json', 'w') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=4)
