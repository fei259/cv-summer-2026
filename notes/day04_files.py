from pathlib import Path

def read_scores(file_path):
    records=[]
    
    with open(file_path,"r",encoding="utf-8") as file:
        for line in file:
            name,score=line.split()
            records.append([name,float(score)])
            
    return records

def calculate_average(records):
    total=0
    
    for name,score in records:
        total+=score
        
    return total/len(records)

def save_result(file_path,records,average):
    with open(file_path,"w",encoding="utf-8") as file:
        file.write(f"人数：{len(records)}\n")
        file.write("成绩记录：\n")
        
        for name,score in records:
            file.write(f"{name} {score}分\n")
            
        file.write(f"平均分： {average:.2f}\n")
        
def main():
    current_dir=Path(__file__).parent       #__file__ 是 Python 内置变量，表示当前正在运行的 Python 文件的路径
    input_path=current_dir/"scores.txt"
    output_path=current_dir/"score_result.txt"
    
    try:
        records = read_scores(input_path)
        average = calculate_average(records)
        save_result(output_path, records, average)
        print(f"共读取 {len(records)} 名学生")
        print(f"平均分：{average:.2f}")
        print(f"结果已保存到 {output_path.name}")
        
    except FileNotFoundError:
        print("错误：找不到成绩文件\n")
        
    except ValueError:
        print("错误：成绩格式不正确\n")
        
if __name__=="__main__":
    main()