#基于的基础镜像
FROM python:3.8

# 制作者信息
MAINTAINER 1421325491<Johnny X.Q. Liao>@qq.com

#代码添加到code文件夹
ADD /Desktop/HealthCondition /usr/src/app

# 设置app文件夹是工作目录
WORKDIR /usr/src/app

# 安装支持
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "/usr/src/app/pca.py" ]