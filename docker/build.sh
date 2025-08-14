#!/bin/bash
# FastText Serving Docker 构建脚本

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
IMPLEMENTATION=""
TAG="latest"
PUSH=false
REGISTRY=""

# 帮助信息
show_help() {
    echo "FastText Serving Docker 构建脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -i, --implementation IMPL  指定实现类型: rust, python, 或 all (默认: all)"
    echo "  -t, --tag TAG             镜像标签 (默认: latest)"
    echo "  -p, --push               构建后推送到镜像仓库"
    echo "  -r, --registry REGISTRY   镜像仓库地址"
    echo "  -h, --help               显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -i python -t v1.0.0                    # 只构建Python版本"
    echo "  $0 -i rust -t v1.0.0 -p -r my-registry   # 构建Rust版本并推送"
    echo "  $0 -i all -t latest                       # 构建所有版本"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--implementation)
            IMPLEMENTATION="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知参数 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 验证实现类型
if [[ -n "$IMPLEMENTATION" && "$IMPLEMENTATION" != "rust" && "$IMPLEMENTATION" != "python" && "$IMPLEMENTATION" != "all" ]]; then
    echo -e "${RED}错误: 实现类型必须是 rust, python 或 all${NC}"
    exit 1
fi

# 默认构建所有实现
if [[ -z "$IMPLEMENTATION" ]]; then
    IMPLEMENTATION="all"
fi

# 设置镜像名称
if [[ -n "$REGISTRY" ]]; then
    IMAGE_PREFIX="${REGISTRY}/fasttext-serving"
else
    IMAGE_PREFIX="fasttext-serving"
fi

# 切换到项目根目录
cd "$(dirname "$0")/.."

echo -e "${BLUE}FastText Serving Docker 构建${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "实现类型: ${YELLOW}$IMPLEMENTATION${NC}"
echo -e "标签: ${YELLOW}$TAG${NC}"
echo -e "推送: ${YELLOW}$PUSH${NC}"
if [[ -n "$REGISTRY" ]]; then
    echo -e "仓库: ${YELLOW}$REGISTRY${NC}"
fi
echo ""

# 构建函数
build_image() {
    local impl=$1
    local dockerfile="docker/${impl}.Dockerfile"
    local image_name="${IMAGE_PREFIX}-${impl}:${TAG}"
    
    echo -e "${GREEN}构建 ${impl} 实现...${NC}"
    echo -e "镜像名: ${YELLOW}$image_name${NC}"
    echo -e "Dockerfile: ${YELLOW}$dockerfile${NC}"
    
    if [[ ! -f "$dockerfile" ]]; then
        echo -e "${RED}错误: Dockerfile 不存在: $dockerfile${NC}"
        return 1
    fi
    
    # 构建镜像
    docker build -f "$dockerfile" -t "$image_name" .
    
    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}✅ ${impl} 镜像构建成功${NC}"
        
        # 推送镜像
        if [[ "$PUSH" == "true" ]]; then
            echo -e "${YELLOW}推送镜像: $image_name${NC}"
            docker push "$image_name"
            if [[ $? -eq 0 ]]; then
                echo -e "${GREEN}✅ ${impl} 镜像推送成功${NC}"
            else
                echo -e "${RED}❌ ${impl} 镜像推送失败${NC}"
                return 1
            fi
        fi
    else
        echo -e "${RED}❌ ${impl} 镜像构建失败${NC}"
        return 1
    fi
    
    echo ""
}

# 执行构建
if [[ "$IMPLEMENTATION" == "all" ]]; then
    echo -e "${BLUE}构建所有实现...${NC}"
    build_image "python"
    build_image "rust"
elif [[ "$IMPLEMENTATION" == "python" ]]; then
    build_image "python"
elif [[ "$IMPLEMENTATION" == "rust" ]]; then
    build_image "rust"
fi

echo -e "${GREEN}🎉 构建完成！${NC}"
echo ""
echo -e "${BLUE}启动命令示例:${NC}"
if [[ "$IMPLEMENTATION" == "all" || "$IMPLEMENTATION" == "python" ]]; then
    echo -e "${YELLOW}Python版本:${NC}"
    echo "  docker run -p 8000:8000 -v /path/to/model:/app/models ${IMAGE_PREFIX}-python:${TAG} --model /app/models/model.bin"
fi
if [[ "$IMPLEMENTATION" == "all" || "$IMPLEMENTATION" == "rust" ]]; then
    echo -e "${YELLOW}Rust版本:${NC}"
    echo "  docker run -p 8000:8000 -v /path/to/model:/app/models ${IMAGE_PREFIX}-rust:${TAG} --model /app/models/model.bin"
fi
