## 合取范式 (CNF) 转化

给定的前提为：
1. \( P \rightarrow Q \)
2. \( \neg P \rightarrow R \)
3. \( Q \vee R \rightarrow S \)

转化为子句形式：

1. \( P \rightarrow Q \)   
   使用第一步，转换为：  
   \( \neg P \vee Q \)

2. \( \neg P \rightarrow R \)  
   使用第一步，转换为：  
   \( P \vee R \)

3. \( Q \vee R \rightarrow S \)  
   使用第一步，转换为：  
   \( \neg(Q \vee R) \vee S \)  
   使用德摩根定律，进一步转换为：  
   \( (\neg Q \wedge \neg R) \vee S \)  
   使用分配律，得到：  
   \( (\neg Q \vee S) \wedge (\neg R \vee S) \)

## 使用归结法

为了证明结论，我们首先得到以下的子句形式：

1. \( \neg P \vee Q \) 
2. \( P \vee R \)
3. \( \neg Q \vee S \)
4. \( \neg R \vee S \)
5. \( \neg S \) （结论的否定）

归结步骤：

a) 从3和5中，我们得到 \( \neg Q \)
即，\( \neg Q \vee S \) 和 \( \neg S \) 归结为 \( \neg Q \)

b) 从4和5中，我们得到 \( \neg R \)
即，\( \neg R \vee S \) 和 \( \neg S \) 归结为 \( \neg R \)

c) 使用a中得到的 \( \neg Q \) 和1，我们有 \( \neg P \)
即，\( \neg P \vee Q \) 和 \( \neg Q \) 归结为 \( \neg P \)

d) 使用b中得到的 \( \neg R \) 和2，我们又得到 \( P \)
即，\( P \vee R \) 和 \( \neg R \) 归结为 \( P \)

此时，我们得到了 \( P \) 和 \( \neg P \) 的矛盾，因此结论S被证明。
