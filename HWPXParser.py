import xml.etree.ElementTree as ET
from pathlib import Path
import logging
import re
from typing import Dict, List, Optional, Union
from bs4 import BeautifulSoup

class HWPXParser:
    def __init__(self, hwpx_path: str):
        self.hwpx_path = Path(hwpx_path)
        self.setup_logging()
        self.content = None
        self.images = []
        self.tables = []

    def setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def parse_hwpx(self) -> None:
        """HWPX 파일 파싱"""
        try:
            # HWPX 파일을 XML로 파싱
            tree = ET.parse(self.hwpx_path)
            root = tree.getroot()
            
            # 네임스페이스 처리
            namespaces = {'owpml': 'http://www.hancom.co.kr/hwpml/2016/format'}
            
            # 문서 내용 파싱
            self.content = self._parse_content(root, namespaces)
            
        except ET.ParseError as e:
            self.logger.error(f"HWPX 파싱 중 XML 오류 발생: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"HWPX 파싱 중 오류 발생: {str(e)}")
            raise

    def _parse_content(self, root: ET.Element, ns: Dict[str, str]) -> List[Dict]:
        """문서 내용 파싱"""
        content = []
        
        # 섹션별로 처리
        for section in root.findall('.//owpml:section', ns):
            section_content = self._parse_section(section, ns)
            content.extend(section_content)
            
        return content

    def _parse_section(self, section: ET.Element, ns: Dict[str, str]) -> List[Dict]:
        """섹션 파싱"""
        section_content = []
        
        for para in section.findall('.//owpml:p', ns):
            para_content = self._parse_paragraph(para, ns)
            if para_content:
                section_content.append(para_content)
                
        return section_content

    def _parse_paragraph(self, para: ET.Element, ns: Dict[str, str]) -> Optional[Dict]:
        """문단 파싱"""
        try:
            para_content = {
                'type': 'paragraph',
                'content': '',
                'style': self._parse_style(para),
            }
            
            # 텍스트 컨텐츠 파싱
            for run in para.findall('.//owpml:run', ns):
                text = run.findtext('.//owpml:text', default='', namespaces=ns)
                para_content['content'] += text
            
            # 이미지 파싱
            images = para.findall('.//owpml:image', ns)
            if images:
                para_content['images'] = [self._parse_image(img) for img in images]
            
            # 표 파싱
            tables = para.findall('.//owpml:table', ns)
            if tables:
                para_content['tables'] = [self._parse_table(table, ns) for table in tables]
            
            return para_content
            
        except Exception as e:
            self.logger.warning(f"문단 파싱 중 오류 발생: {str(e)}")
            return None

    def _parse_style(self, elem: ET.Element) -> Dict:
        """스타일 정보 파싱"""
        style = {}
        
        # 기본 스타일 속성들
        style_attrs = [
            'fontSize', 'fontName', 'bold', 'italic', 'underline',
            'textAlign', 'lineSpacing', 'backgroundColor', 'color'
        ]
        
        for attr in style_attrs:
            value = elem.get(attr)
            if value:
                style[attr] = value
                
        return style

    def _parse_image(self, img_elem: ET.Element) -> Dict:
        """이미지 정보 파싱"""
        image_info = {
            'type': 'image',
            'id': img_elem.get('id', ''),
            'width': img_elem.get('width', ''),
            'height': img_elem.get('height', ''),
            'format': img_elem.get('format', '')
        }
        
        # 이미지 데이터나 경로 처리
        binary_data = img_elem.find('.//owpml:binary-data')
        if binary_data is not None:
            image_info['data'] = binary_data.text
            
        return image_info

    def _parse_table(self, table_elem: ET.Element, ns: Dict[str, str]) -> Dict:
        """표 파싱"""
        table_data = {
            'type': 'table',
            'rows': [],
            'style': self._parse_style(table_elem)
        }
        
        for row in table_elem.findall('.//owpml:tr', ns):
            row_data = []
            for cell in row.findall('.//owpml:td', ns):
                cell_content = {
                    'content': cell.findtext('.//owpml:text', default='', namespaces=ns),
                    'style': self._parse_style(cell)
                }
                row_data.append(cell_content)
            table_data['rows'].append(row_data)
            
        return table_data

    def to_html(self, output_path: str) -> None:
        """파싱된 내용을 HTML로 변환"""
        try:
            self.parse_hwpx()
            
            html_content = [
                '<!DOCTYPE html>',
                '<html>',
                '<head>',
                '<meta charset="UTF-8">',
                '<style>',
                self._generate_css(),
                '</style>',
                '</head>',
                '<body>'
            ]
            
            # 컨텐츠 변환
            for item in self.content:
                html_content.append(self._content_to_html(item))
                
            html_content.extend(['</body>', '</html>'])
            
            # HTML 파일 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(html_content))
                
            self.logger.info(f"HTML 파일이 성공적으로 생성되었습니다: {output_path}")
            
        except Exception as e:
            self.logger.error(f"HTML 변환 중 오류 발생: {str(e)}")
            raise

    def _content_to_html(self, content: Dict) -> str:
        """컨텐츠를 HTML로 변환"""
        if content['type'] == 'paragraph':
            return self._paragraph_to_html(content)
        elif content['type'] == 'table':
            return self._table_to_html(content)
        elif content['type'] == 'image':
            return self._image_to_html(content)
        return ''

    def _paragraph_to_html(self, para: Dict) -> str:
        """문단을 HTML로 변환"""
        style = self._style_to_css(para['style'])
        html = f'<div class="paragraph" style="{style}">'
        
        # 텍스트 내용
        if para['content']:
            html += f'<p>{para["content"]}</p>'
            
        # 이미지
        if 'images' in para:
            for img in para['images']:
                html += self._image_to_html(img)
                
        # 표
        if 'tables' in para:
            for table in para['tables']:
                html += self._table_to_html(table)
                
        html += '</div>'
        return html

    def _table_to_html(self, table: Dict) -> str:
        """표를 HTML로 변환"""
        style = self._style_to_css(table['style'])
        html = f'<table style="{style}">'
        
        for row in table['rows']:
            html += '<tr>'
            for cell in row:
                cell_style = self._style_to_css(cell['style'])
                html += f'<td style="{cell_style}">{cell["content"]}</td>'
            html += '</tr>'
            
        html += '</table>'
        return html

    def _image_to_html(self, image: Dict) -> str:
        """이미지를 HTML로 변환"""
        if 'data' in image:
            # base64 인코딩된 이미지 데이터가 있는 경우
            return f'<img src="data:image/{image["format"]};base64,{image["data"]}" ' \
                   f'width="{image["width"]}" height="{image["height"]}">'
        else:
            # 이미지 참조만 있는 경우
            return f'<img src="{image["id"]}" ' \
                   f'width="{image["width"]}" height="{image["height"]}">'

    def _style_to_css(self, style: Dict) -> str:
        """스타일 정보를 CSS로 변환"""
        css_parts = []
        
        # 스타일 속성 매핑
        style_mapping = {
            'fontSize': 'font-size',
            'fontName': 'font-family',
            'bold': 'font-weight',
            'italic': 'font-style',
            'underline': 'text-decoration',
            'textAlign': 'text-align',
            'lineSpacing': 'line-height',
            'backgroundColor': 'background-color',
            'color': 'color'
        }
        
        for key, value in style.items():
            if key in style_mapping:
                css_key = style_mapping[key]
                if key == 'bold' and value == 'true':
                    css_parts.append('font-weight: bold')
                elif key == 'italic' and value == 'true':
                    css_parts.append('font-style: italic')
                elif key == 'underline' and value == 'true':
                    css_parts.append('text-decoration: underline')
                else:
                    css_parts.append(f'{css_key}: {value}')
                    
        return '; '.join(css_parts)

    def _generate_css(self) -> str:
        """기본 CSS 스타일 생성"""
        return """
            body {
                font-family: 'Malgun Gothic', sans-serif;
                line-height: 1.6;
                margin: 2em;
            }
            .paragraph {
                margin-bottom: 1em;
            }
            table {
                border-collapse: collapse;
                margin: 1em 0;
                width: 100%;
            }
            td {
                border: 1px solid #ddd;
                padding: 8px;
            }
            img {
                max-width: 100%;
                height: auto;
                margin: 1em 0;
            }
        """

def convert_hwpx_to_html(input_path: str, output_path: str) -> None:
    """HWPX 파일을 HTML로 변환하는 편의 함수"""
    parser = HWPXParser(input_path)
    parser.to_html(output_path)

--------------------------------------------------------------------------------
import olefile
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from typing import Dict, List, Optional
import io

class HWPXParser:
    def __init__(self, hwpx_path: str):
        self.hwpx_path = Path(hwpx_path)
        self.setup_logging()
        self.content = None
        self.ole = None

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def parse_hwpx(self):
        """HWPX(CFB) 파일 파싱"""
        try:
            # OLE 파일 열기
            self.ole = olefile.OleFileIO(self.hwpx_path)
            
            # 스트림 목록 확인
            streams = self.ole.listdir()
            self.logger.info(f"Found streams: {streams}")
            
            # 주요 스트림 읽기
            if ['FileHeader'] in streams:
                header = self.ole.openstream('FileHeader').read()
                self.logger.info(f"FileHeader content: {header[:100]}")
                
            if ['BodyText', 'Section0'] in streams:
                section = self.ole.openstream(['BodyText', 'Section0']).read()
                self.logger.info(f"Section0 content: {section[:100]}")
                
            # 기타 스트림들도 필요에 따라 처리
            
        except Exception as e:
            self.logger.error(f"HWPX 파싱 중 오류 발생: {str(e)}")
            raise
        finally:
            if self.ole:
                self.ole.close()

def inspect_hwpx(file_path: str) -> None:
    """HWPX 파일 구조 검사"""
    parser = HWPXParser(file_path)
    parser.parse_hwpx()


# 사용
inspect_hwpx(input_path)

-----------------------------------------------------------------------------------
from pathlib import Path
import logging
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET
import struct

class HWPXParser:
    """
    HWPX 파일 파서
    HWPX는 HWP의 각 컴포넌트를 OWPML로 표현한 XML 기반 포맷
    """
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.setup_logging()
        self._init_hwp_constants()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _init_hwp_constants(self):
        """HWP 문서에서 사용되는 상수들 정의"""
        # 단위 변환 (HWP 단위 -> 포인트)
        self.HWPUNIT_TO_POINT = 0.352778

        # 문단 정렬
        self.ALIGN_TYPES = {
            0: 'left',
            1: 'right', 
            2: 'center',
            3: 'justify'
        }

        # 텍스트 장식
        self.TEXT_DECORATIONS = {
            1: 'underline',
            2: 'strike',
            3: 'overline'
        }

    def parse_file(self):
        """HWPX 파일 파싱 시작"""
        try:
            self.logger.debug(f"파싱 시작: {self.file_path}")
            
            # 파일 내용 읽기
            with open(self.file_path, 'rb') as f:
                content = f.read()
            
            # 파일 시그니처와 구조 확인을 위해 처음 100바이트 출력
            self.logger.debug("파일 시그니처:")
            self.logger.debug(' '.join(f'{b:02x}' for b in content[:100]))
            
            # 텍스트로도 출력
            try:
                text_content = content[:100].decode('utf-8', errors='replace')
                self.logger.debug(f"텍스트 형태: {text_content}")
            except UnicodeDecodeError:
                self.logger.debug("텍스트 디코딩 실패")

            # 파일 크기
            self.logger.debug(f"파일 크기: {len(content)} bytes")

            return self._parse_content(content)

        except Exception as e:
            self.logger.error(f"파싱 중 오류 발생: {str(e)}")
            raise

    def _parse_content(self, content: bytes) -> Dict:
        """파일 내용 파싱"""
        # 여기서 실제 파싱 로직 구현
        # 현재는 파일의 구조를 파악하기 위한 기본 정보만 반환
        return {
            'size': len(content),
            'header': content[:100].hex(),
            'signature': ' '.join(f'{b:02x}' for b in content[:8])
        }

def analyze_hwpx(file_path: str) -> Dict:
    """
    HWPX 파일 분석을 위한 헬퍼 함수
    파일의 구조와 내용을 분석하여 정보 반환
    """
    parser = HWPXParser(file_path)
    return parser.parse_file()

-----------------------------------------------------------------------------------------
import xml.etree.ElementTree as ET
import logging
from pathlib import Path
import olefile
from typing import Dict, Union, List, Optional

class HWPXConverter:
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def detect_file_type(self) -> str:
        """
        파일 타입 감지
        Returns:
            'xml': XML 기반 HWPX
            'binary': 바이너리 형식의 HWPX
        """
        try:
            # 먼저 XML로 파싱 시도
            ET.parse(self.input_path)
            return 'xml'
        except ET.ParseError:
            # XML 파싱 실패시 바이너리로 가정
            try:
                with open(self.input_path, 'rb') as f:
                    signature = f.read(8)
                    # HWP 바이너리 시그니처 확인
                    if signature == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
                        return 'binary'
                    else:
                        raise ValueError("지원하지 않는 파일 형식입니다.")
            except Exception as e:
                self.logger.error(f"파일 타입 감지 중 오류 발생: {str(e)}")
                raise

    def parse_xml_hwpx(self) -> Dict:
        """XML 기반 HWPX 파싱"""
        try:
            tree = ET.parse(self.input_path)
            root = tree.getroot()
            
            # OWPML 네임스페이스 처리
            ns = {'hwpx': 'http://www.hancom.co.kr/hwpml/2016/format'}
            
            content = {
                'text': [],
                'images': [],
                'tables': []
            }
            
            # 섹션별 처리
            for section in root.findall('.//hwpx:section', ns):
                # 텍스트 추출
                for para in section.findall('.//hwpx:p', ns):
                    text = self._extract_text_from_para(para, ns)
                    if text:
                        content['text'].append(text)
                
                # 이미지 추출
                for img in section.findall('.//hwpx:image', ns):
                    img_data = self._extract_image_data(img, ns)
                    if img_data:
                        content['images'].append(img_data)
                
                # 표 추출
                for table in section.findall('.//hwpx:table', ns):
                    table_data = self._extract_table_data(table, ns)
                    if table_data:
                        content['tables'].append(table_data)
            
            return content
            
        except Exception as e:
            self.logger.error(f"XML HWPX 파싱 중 오류 발생: {str(e)}")
            raise

    def parse_binary_hwpx(self) -> Dict:
        """바이너리 형식의 HWPX 파싱"""
        try:
            ole = olefile.OleFileIO(self.input_path)
            
            content = {
                'text': [],
                'images': [],
                'tables': []
            }
            
            # 텍스트 섹션 처리
            if ole.exists('BodyText/Section0'):
                section = ole.openstream('BodyText/Section0').read()
                content['text'] = self._process_binary_section(section)
            
            # 이미지 처리 (BinData 스트림)
            for entry in ole.listdir():
                if entry[0] == 'BinData':
                    img_data = self._process_binary_image(ole, entry)
                    if img_data:
                        content['images'].append(img_data)
            
            ole.close()
            return content
            
        except Exception as e:
            self.logger.error(f"바이너리 HWPX 파싱 중 오류 발생: {str(e)}")
            raise

    def to_html(self, output_path: str) -> None:
        """HWPX를 HTML로 변환"""
        try:
            file_type = self.detect_file_type()
            
            if file_type == 'xml':
                content = self.parse_xml_hwpx()
            else:  # binary
                content = self.parse_binary_hwpx()
            
            # HTML 생성
            html_content = self._generate_html(content)
            
            # 파일 저장
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"HTML 파일 생성 완료: {output_path}")
            
        except Exception as e:
            self.logger.error(f"HTML 변환 중 오류 발생: {str(e)}")
            raise

    def _extract_text_from_para(self, para: ET.Element, ns: Dict) -> str:
        """문단에서 텍스트 추출"""
        text_parts = []
        for run in para.findall('.//hwpx:run', ns):
            text = run.findtext('.//hwpx:text', default='', namespaces=ns)
            if text:
                text_parts.append(text)
        return ' '.join(text_parts)

    def _extract_image_data(self, img: ET.Element, ns: Dict) -> Dict:
        """이미지 데이터 추출"""
        return {
            'id': img.get('id', ''),
            'width': img.get('width', ''),
            'height': img.get('height', ''),
            'format': img.get('format', '')
        }

    def _extract_table_data(self, table: ET.Element, ns: Dict) -> List[List[str]]:
        """표 데이터 추출"""
        table_data = []
        for row in table.findall('.//hwpx:tr', ns):
            row_data = []
            for cell in row.findall('.//hwpx:td', ns):
                text = cell.findtext('.//hwpx:text', default='', namespaces=ns)
                row_data.append(text)
            table_data.append(row_data)
        return table_data

    def _process_binary_section(self, section: bytes) -> List[str]:
        """바이너리 섹션 처리"""
        # HWP 바이너리 형식에 맞춰 텍스트 추출
        # 실제 구현시 HWP 파일 형식 명세 필요
        # 여기서는 간단한 예시만 구현
        texts = []
        # 바이너리 parsing 로직 구현
        return texts

    def _process_binary_image(self, ole: olefile.OleFileIO, entry: List[str]) -> Optional[Dict]:
        """바이너리 이미지 처리"""
        try:
            with ole.openstream(entry) as stream:
                image_data = stream.read()
                return {
                    'id': entry[-1],
                    'data': image_data,
                    'format': entry[-1].split('.')[-1].lower()
                }
        except Exception as e:
            self.logger.warning(f"이미지 처리 중 오류 발생: {str(e)}")
            return None

    def _generate_html(self, content: Dict) -> str:
        """HTML 생성"""
        html = ['<!DOCTYPE html>', '<html>', '<head>',
                '<meta charset="UTF-8">',
                '<style>', self._generate_css(), '</style>',
                '</head>', '<body>']
        
        # 텍스트 처리
        for text in content['text']:
            html.append(f'<p>{text}</p>')
        
        # 이미지 처리
        for img in content['images']:
            if 'data' in img:
                # 바이너리 데이터가 있는 경우
                import base64
                img_data = base64.b64encode(img['data']).decode('utf-8')
                html.append(
                    f'<img src="data:image/{img["format"]};base64,{img_data}" '
                    f'alt="Image {img["id"]}">'
                )
            else:
                # 이미지 참조만 있는 경우
                html.append(
                    f'<img src="{img["id"]}" '
                    f'width="{img["width"]}" height="{img["height"]}" '
                    f'alt="Image {img["id"]}">'
                )
        
        # 표 처리
        for table in content['tables']:
            table_html = ['<table>']
            for row in table:
                table_html.append('<tr>')
                for cell in row:
                    table_html.append(f'<td>{cell}</td>')
                table_html.append('</tr>')
            table_html.append('</table>')
            html.append('\n'.join(table_html))
        
        html.extend(['</body>', '</html>'])
        return '\n'.join(html)

    def _generate_css(self) -> str:
        """기본 CSS 스타일 생성"""
        return """
            body {
                font-family: 'Malgun Gothic', sans-serif;
                line-height: 1.6;
                margin: 2em;
            }
            p {
                margin: 1em 0;
            }
            table {
                border-collapse: collapse;
                margin: 1em 0;
                width: 100%;
            }
            td {
                border: 1px solid #ddd;
                padding: 8px;
            }
            img {
                max-width: 100%;
                height: auto;
                margin: 1em 0;
            }
        """

def convert_to_html(input_path: str, output_path: str) -> None:
    """HWPX 파일을 HTML로 변환하는 편의 함수"""
    converter = HWPXConverter(input_path)
    converter.to_html(output_path)

----------------------------------------------------------------------------------
from docling.document_converter import DocumentConverter

source = "./output_images/result.html"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"

----------------------------------------------------------------------------------

import easyocr
import os
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter

output_path = './output_images'
input_path = './hwpx/혜안 고도화 사업 완료보고 및 신규 서비스 시연 계획.hwpx'

from hwp5.hwp5html import main as hwp5html_main
import sys

def convert_hwpx_to_html(hwp_file, output_file):
    # 원래의 sys.argv 저장
    original_argv = sys.argv
    
    # hwp5html의 올바른 인수 형식으로 변경
    sys.argv = ['hwp5html', hwp_file, '--output', output_file]
    
    try:
        # 변환 실행
        hwp5html_main()
    finally:
        # sys.argv 복원
        sys.argv = original_argv

convert_hwpx_to_html(input_path, output_path)

def extract_image_data_with_context(html_path):
    """Extract image paths and their positions in the HTML structure."""
    with open(html_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    image_data = []
    for img in soup.find_all("img"):
        # Get image src and parent element for context
        src = img.get("src")
        parent = img.find_parent()  # Find the parent tag of the image
        if src:
            image_data.append({"src": src, "parent": parent, "tag": img})
    return image_data
  
def perform_ocr_and_map_with_easyocr(image_dir, image_data):
    """
    Perform OCR on all images using EasyOCR and map the results with their context.
    """
    reader = easyocr.Reader(['ko'], gpu=False)  # Initialize EasyOCR reader (add languages as needed)
    ocr_results = []

    for img_info in image_data:
        img_path = os.path.join(image_dir, os.path.basename(img_info["src"]))
        try:
            # Perform OCR
            results = reader.readtext(img_path, detail=0)  # Extract text without positional details
            text = " ".join(results).strip()  # Join all OCR results into a single string

            ocr_results.append({"src": img_info["src"], "text": text, "parent": img_info["parent"]})
        except Exception as e:
            ocr_results.append({"src": img_info["src"], "text": f"Error processing {img_path}: {e}", "parent": img_info["parent"]})

    return ocr_results

from lxml import html
import lxml.etree as etree

def xhtml_to_html(xhtml_file, html_file):
    # XHTML 파일 파싱
    tree = etree.parse(xhtml_file)
    
    # HTML로 변환
    html_content = etree.tostring(tree, pretty_print=True, method='html')
    
    # 파일로 저장
    with open(html_file, 'wb') as f:
        f.write(html_content)

xhtml_to_html("./output_images/index.xhtml", "./output_images/result.html")
html_path = "./output_images/result.html"
image_dir = "./output_images/bindata"
final = "./output_images/final_result.html"

image_data = extract_image_data_with_context(html_path)

# Step 2: Perform OCR and map results to their contexts
ocr_results = perform_ocr_and_map_with_easyocr(image_dir, image_data)

# OCR 결과를 HTML 파일에 반영
def update_html_with_ocr(html_content, ocr_results):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    for ocr in ocr_results:
        src = ocr['src']
        text = ocr['text']
        parent_html = str(ocr['parent'])  # HTML 문자열로 변환
        
        # 부모 HTML을 파싱
        parent_soup = BeautifulSoup(parent_html, 'html.parser')
        
        # <img> 태그를 찾기
        img_tag = parent_soup.find('img', {'src': src})
        
        if img_tag:
            # <img> 태그를 텍스트로 대체
            img_tag.replace_with(BeautifulSoup(f"<p>{text}</p>", 'html.parser'))
        
        # 원래의 부모 HTML을 텍스트로 변환
        parent_html_updated = str(parent_soup)
        
        # 원래 문서에서 해당 <img> 태그의 부모 요소를 찾아 수정
        original_parent_tag = soup.find(attrs={"src": src}).find_parent()
        if original_parent_tag:
            original_parent_tag.replace_with(BeautifulSoup(parent_html_updated, 'html.parser'))
    
    return str(soup)

# HTML 파일 로드
with open(html_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

# 업데이트된 HTML 생성
updated_html = update_html_with_ocr(html_content, ocr_results)

# HTML 파일 저장
with open(final, 'w', encoding='utf-8') as file:
    file.write(updated_html)

def docling_parsing(source):
    source = source
    converter = DocumentConverter()
    result = converter.convert(source)
    return result

def docling_parsing(source):
    source = source
    converter = DocumentConverter()
    result = converter.convert(source)
    return result

source = final  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)

result = docling_parsing(source)

print(result.document.export_to_markdown())

