from typing import List


class Simulator:
    def load_templates(self, templates: List[dict]):
        """
        주어진 템플릿 설정의 list를 읽어 template을 등록합니다.

        :param List[dict] templates: 각 template의 설정값
        """

    def set_seed(self, seed: int):
        """
        난수생성기의 초기값을 seed로 초기화합니다.

        :param int seed: 초기값
        """

    def generate(self, n: int) -> List[List[Problem]]:
        """
        각 template마다 n개의 새로운 문제를 생성합니다.

        :param int n: template 당 생성될 문제의 개수
        """
