import json
from pathlib import Path

class JsonFileManager:
    def __init__(self, filename):
        self.filename = filename

    def read_json(self):
        """JSON 파일에서 데이터를 읽어서 Python 사전으로 반환합니다."""
        try:
            with open(self.filename, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            return {}  # 파일이 존재하지 않을 경우 빈 사전 반환
        except json.JSONDecodeError:
            return {}  # JSON 파일이 잘못되었을 경우 빈 사전 반환

    def write_json(self, data):
        """Python 사전을 JSON 파일에 쓰기."""
        with open(self.filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def write_object_as_json(self, obj):
        """특정 객체의 상태를 JSON으로 변환하여 파일에 저장합니다."""
        try:
            obj_dict = obj.__dict__  # 객체의 __dict__를 사용하여 속성을 사전으로 변환
            self.write_json(obj_dict)
        except AttributeError:
            raise ValueError("Provided object is not serializable")

# 사용 예
class SampleClass:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# 인스턴스 생성 및 JSON 파일 작성
sample = SampleClass("John Doe", 30)
manager = JsonFileManager("sample.json")
manager.write_object_as_json(sample)
