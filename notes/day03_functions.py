def calculate_average(numbers):
    if len(numbers)==0:
        return None
    
    total=0
    
    for num in numbers:
        total+=num
        
    return total/len(numbers)




def count_words(words):
    counts={}
    
    for word in words:
        counts[word]=counts.get(word,0)+1
        
    return counts




def query_score(students,name):
    if name not in students:
        return None
    
    return students[name]




def show_menu():
    print("""
===== 学生成绩管理 =====
1. 查询成绩
2. 修改成绩
3. 显示全部
0. 退出""")
    
def main():
    students={
        "张三":92,
        "李四":85,
        "王五":78
    }
    
    while True:
        show_menu()
        choice=input("请输入操作类型：")
        
        if choice == "1":
            name = input("请输入要查询的姓名：")
            score = query_score(students, name)

            if score is None:
                print("没有找到该学生")
            else:
                print(f"{name}的成绩为：{score}")

        elif choice == "2":
            name = input("请输入要修改的姓名：")

            if name not in students:
                print("没有找到该学生")
            else:
                score = int(input("请输入新成绩："))
                students[name] = score
                print("修改成功")

        elif choice == "3":
            for name, score in students.items():
                print(name, score)

        elif choice == "0":
            print("程序已退出")
            break

        else:
            print("输入操作不合法！")
                    
if __name__=="__main__":
    main()
            