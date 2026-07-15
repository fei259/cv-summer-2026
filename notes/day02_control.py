#练习1：成绩分段
score=int(input("请输入您的成绩："))

if 90<=score<=100:
    print("优秀")
elif 80<=score<=89:
    print("良好")
elif 60<=score<=79:
    print("及格")
elif 0<=score<=59:
    print("不及格")
else:
    print("成绩不合法")
    
    
    
#猜数字
ANSWER=37

while True:
    num=int(input("请输入数字："))
    
    if num>37:
        print("猜大了")
    elif num<37:
        print("猜小了")
    else:
        print("猜对了")
        break
    
    
    
#手动找最大值和最小值
numbers=[18, 3, 42, -5, 27, 9]
maxinum=numbers[0]
mininum=numbers[0]

for num in numbers:
    if num>maxinum:
        maxinum=num
    
    if num<mininum:
        mininum=num
        
print("最大值：", maxinum)
print("最小值：", mininum)
        
        
        
        
#列表去重
numbers = [1, 2, 1, 3, 2, 4, 3]
result=[]

for num in numbers:
    if num not in result:
        result.append(num)
        
print(result)
        
        
        
        
        
#简单列表菜单
# 循环显示：
# 1. 添加
# 2. 删除
# 3. 修改
# 4. 查询
# 5. 显示全部
# 0. 退出
items=[]
while True:
    print("""
===== 待办事项管理器 =====
1. 添加事项
2. 删除事项
3. 修改事项
4. 查询事项
5. 显示全部
0. 退出
""")
    choice=input("请输入操作：")

    if choice=='1':
        item=input("请输入要添加的事项：")
        items.append(item)
    elif choice=='2':
        if not items:
            print("当前没有待办事项")
            
            continue
            
        index=int(input("请输入要删除的事项编号："))
        
        if 1<=index<=len(items):
            items.pop(index-1)
        else:
            print("输入编号不合法")
    elif choice=='3':
        if not items:
            print("当前没有待办事项")
            
            continue
            
        index=int(input("请输入要修改的事项编号："))
        
        if 1<=index<=len(items):
            item=input("请输入修改后的事项：")
            items[index-1]=item
        else:
            print("输入编号不合法")
    elif choice=='4':
        keyword=input("请输入查询关键词：")
        found=False
        
        for index, item in enumerate(items, start=1):       #start=1表示产生的编号从1开始
            if keyword in item:
                print(index, item)
                found = True

        if not found:
            print("没有找到相关事项")
    elif choice=='5':
        if not items:
            print("当前没有待办事项")
            
            continue
            
        print("完整列表菜单如下：\n")
        
        for index,item in enumerate(items):
            print(f"事项编号：{index+1} 事项：{item}\n")
    elif choice=='0':
        print("程序已退出")
        
        break
    else:
        print("该操作不合法")
    

    
        

