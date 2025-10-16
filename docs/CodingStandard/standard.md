# Project_Shopee

## Shopee Standard

_본 스텐다드는 [Ros2 Jazzy 공식문서](https://docs.ros.org/en/jazzy/The-ROS2-Project/Contributing/Code-Style-Language-Versions.html), [Google Style Guides](https://google.github.io/styleguide/)를 참조하여 작성되었습니다._

### Global Standard

__1. 세로 간격 및 빈 줄 규칙__

| 구분             | 간격 |
| ---------------- | ---- |
| 헤더와 본문 사이 | 2줄  |
| 블록 사이        | 1줄  |
| 함수와 함수사이 | 1줄  |

__2. Import문 규칙__

한줄에 하나의 ``import``문을 사용

__3. 중괄호 & 줄바꿈 규칙__

> __Function, Class, enum, struct__
>
> 1. 여는 중괄호(open Brace) 사용
>
> <!--
>
> ~~~
> void MyFunction()
> {
>    ...body...
> }
>
> class Robot
> {
> public:
>   void Move();
> };
> ~~~
>
> -->
>
> 2. 줄바꿈 규칙
>
>    함수 호출의 인자가 길어 한 줄에 다 들어가지 않을 경우, `<br>`
>    여는 괄호 (``(``) 뒤에서 줄을 바꾸고
>
>    - C++ 의 경우 2칸 들여쓰기
>    - Python 의 경우 4칸(``tab``) 들여쓰기
>
>    이후 인자가 더 이어질 경우에도 같은 들여쓰기 유지

> __제어문__
>
> 1. 제어문의 본문이 한 줄이라도 __반드시__ 중괄호 ``{}`` 사용
> 2. 제어문에서는 조건식은 붙인 중괄호(cuddled brace)를 사용
> 3. 단, 조건문이 길어서 줄바꿈이 필요할 경우 조건문이 끝난 뒤 여는 중괄호(open Brace) 사용

__4. 따옴표 규칙__

문자열 내에 작은따옴표가 존재하지 않는 한 작은 따옴표 사용

### Ros2 Standard

1. Packages Names

   `snake_case`
2. Node/Topic/Service/Action/Parameter Names

   `snake_case`
3. Type Names

   `PascalCase`
4. Type Field Names

   `snake_case`
5. Type Constants Names

   `SCREAMING_SNAKE_CASE`

### C++ Standard

> __기본 규칙__
>
> 1. 목적 또는 의도를 나타내는 이름 사용
> 2. 가로 공간 절약하지 말 것
> 3. 약어 및 이니셜 사용 지양(단, 위키에 등재된 약어는 사용 가능)
> 4. 전역 변수의 경우 구체적인 이름 사용
> 5. 템플릿 매개변수는 해당 범주에 따른 명명 규칙을 따름
>
>    - 타입 템플릿 = 타입
>    - 비타입 템플릿 = 변수 또는 상수

1. File Names

   `snake_case`

2. Type Names

   `PascalCase`

3. Concept Names

   `PascalCase`

4. Function Names

   - 기본적으로 `PascalCase`
   - 접근자(`get`/`set`)는 `snake_case`

5. Variable Names

   - Class Data Members
     - `snake_case` + `_`(underscore)
   - Struct Data Members
     - `snake_case`

6. Constant Names

   `k` + `PascalCase`

   - 정적 저장 기간 변수 = 상수 규칙
   - 자동 저장 기간 변수 = 변수 규칙

7. Enumerator Names

   `k` + `PascalCase`

8. Macro Names

   `SCREAMING_SNAKE_CASE`

9. Namespace Names

   `snake_case`

   __네임스페이스 추가사항__

   - 최상위 네임스페이스 이름은 코드의 프로젝트 또는 팀이름
   - 잘 알려진 최상위 네임스페이스와 이름이 겹치지 않도록 중첩 네임스페이스는 고유한 프로젝트 식별사를 사용
   - 깊은 중첩 네임스페이스는 지양
   - 네임스페이스의 이름은 약어 사용금지

### Python Standard

- Package 및 module 이름: `snake_case`
- Class 및 exception 이름: `PascalCase`
- Function, method, parameter, local/instance/global 변수 이름: `snake_case`
- Global/Class constants: `SCREAMING_SNAKE_CASE`

## Comments

- 주석은 한 줄로 작성
- 주석 언어는 한글로 통일
- `C++`은 `//`만 사용할 것
- `Python`은 `#`만 사용할 것
