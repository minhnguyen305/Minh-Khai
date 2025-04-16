import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

# Kiểm tra và import wordcloud
try:
    from wordcloud import WordCloud
except ImportError:
    st.error("Thư viện 'wordcloud' chưa được cài đặt. Vui lòng chạy lệnh: `pip install wordcloud`")
    st.stop()

# Thiết lập cấu hình trang
st.set_page_config(page_title="Phân tích dữ liệu phim", layout="wide")

# Tiêu đề
st.title("Phân tích dữ liệu phim")

# Hàm kiểm tra dữ liệu rỗng
def check_empty_data(data):
    if data.empty:
        st.error("Không có dữ liệu phù hợp với bộ lọc. Vui lòng điều chỉnh lựa chọn.")
        st.stop()

# Đọc dữ liệu
url = "https://raw.githubusercontent.com/nv-thang/Data-Visualization-Course/main/Dataset%20for%20Practice/movies.csv"
movies_data = pd.read_csv(url)
cleaned_data = movies_data.dropna()
if 'year' not in cleaned_data.columns:
    cleaned_data['year'] = pd.to_datetime(cleaned_data['released'], errors='coerce').dt.year.fillna(1900).astype(int)

# Sidebar - Thiết lập
st.sidebar.title("🎬 Bộ công cụ phân tích")

# Tùy chọn dữ liệu
with st.sidebar.expander("🔍 Tùy chọn dữ liệu", expanded=True):
    data_mode = st.radio("Chọn nguồn dữ liệu:", ["Dữ liệu gốc", "Dữ liệu đã làm sạch"])
    current_data = movies_data if data_mode == "Dữ liệu gốc" else cleaned_data

    # Lọc theo thể loại
    all_genres = ["Tất cả"] + sorted(current_data['genre'].unique().tolist())
    selected_genre = st.selectbox("Lọc theo thể loại:", all_genres)
    if selected_genre != "Tất cả":
        current_data = current_data[current_data['genre'] == selected_genre]

    # Lọc theo điểm đánh giá
    score_range = st.slider(
        "Chọn khoảng điểm đánh giá:",
        min_value=float(current_data['score'].min()),
        max_value=float(current_data['score'].max()),
        value=(float(current_data['score'].min()), float(current_data['score'].max()))
    )
    current_data = current_data[(current_data['score'] >= score_range[0]) & 
                               (current_data['score'] <= score_range[1])]

    # Lọc theo ngân sách
    budget_range = st.slider(
        "Chọn khoảng ngân sách (USD):",
        min_value=float(current_data['budget'].min()),
        max_value=float(current_data['budget'].max()),
        value=(float(current_data['budget'].min()), float(current_data['budget'].max()))
    )
    current_data = current_data[(current_data['budget'] >= budget_range[0]) & 
                               (current_data['budget'] <= budget_range[1])]

check_empty_data(current_data)

# Tùy chỉnh giao diện
with st.sidebar.expander("🎨 Tùy chỉnh giao diện", expanded=False):
    color_theme = st.selectbox("Chọn bảng màu:", ["Mặc định", "Pastel", "Viridis", "Magma", "Cividis"])
    if color_theme == "Mặc định":
        chart_color = st.color_picker("Tùy chỉnh màu:", "#800000")
        chart_palette = None
    else:
        chart_palette = color_theme.lower()
        chart_color = None

    chart_size = st.select_slider("Kích thước biểu đồ:", options=["Nhỏ", "Vừa", "Lớn"], value="Vừa")
    fig_width, fig_height = (8, 4) if chart_size == "Nhỏ" else (12, 6) if chart_size == "Vừa" else (18, 10)

# Thông tin tóm tắt
st.write("### Thông tin về tập dữ liệu")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tổng số phim", len(current_data))
with col2:
    st.metric("Số thể loại", len(current_data['genre'].unique()))
with col3:
    st.metric("Điểm đánh giá TB", round(current_data['score'].mean(), 2))

# Hiển thị dữ liệu
num_rows = st.slider("Số dòng dữ liệu hiển thị:", min_value=1, max_value=min(50, len(current_data)), value=5)
st.dataframe(current_data.head(num_rows))

# Tùy chọn tải dữ liệu hiển thị
if st.button("Tải xuống dữ liệu hiển thị (CSV)"):
    csv = current_data.head(num_rows).to_csv(index=False)
    st.download_button(
        label="Nhấn để tải",
        data=csv,
        file_name="movie_data.csv",
        mime="text/csv"
    )

# Sidebar - Chọn loại phân tích
analysis_type = st.sidebar.selectbox(
    "📊 Chọn loại phân tích:",
    ["Tổng quan", "Ngân sách theo thể loại", "Phân tích theo năm", "Top phim", "Phân tích tương quan", "Phân tích từ khóa", "Thống kê"]
)

# Hàm vẽ biểu đồ và lưu
def plot_and_save(fig, filename="chart.png"):
    st.pyplot(fig)
    fig.savefig(filename)
    with open(filename, "rb") as file:
        st.download_button(
            label="Tải xuống biểu đồ (PNG)",
            data=file,
            file_name=filename,
            mime="image/png"
        )

# Phân tích
if analysis_type == "Tổng quan":
    st.write("## Tổng quan về dữ liệu phim")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phân bố thể loại phim")
        genre_counts = current_data['genre'].value_counts().head(10)
        fig = plt.figure(figsize=(fig_width, fig_height))
        if chart_palette:
            sns.barplot(x=genre_counts.values, y=genre_counts.index, palette=chart_palette)
        else:
            sns.barplot(x=genre_counts.values, y=genre_counts.index, color=chart_color)
        plt.xlabel('Số lượng phim')
        plt.ylabel('Thể loại')
        plt.title('10 thể loại phim phổ biến nhất')
        plt.tight_layout()
        plot_and_save(fig, "genre_distribution.png")
    
    with col2:
        st.subheader("Xu hướng điểm đánh giá theo năm")
        yearly_ratings = current_data.groupby('year')['score'].mean().reset_index()
        fig = plt.figure(figsize=(fig_width, fig_height))
        if chart_palette:
            sns.lineplot(data=yearly_ratings, x='year', y='score', palette=chart_palette)
        else:
            plt.plot(yearly_ratings['year'], yearly_ratings['score'], color=chart_color)
        plt.xlabel('Năm')
        plt.ylabel('Điểm đánh giá trung bình')
        plt.title('Xu hướng điểm đánh giá phim theo năm')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_and_save(fig, "rating_trend.png")

elif analysis_type == "Ngân sách theo thể loại":
    st.write("## Ngân sách trung bình của phim theo thể loại")
    
    with st.sidebar.expander("🔢 Tùy chọn sắp xếp", expanded=True):
        sort_option = st.radio("Sắp xếp theo:", ["Không sắp xếp", "Tăng dần", "Giảm dần"])
        show_avg_line = st.checkbox("Hiển thị đường trung bình", True)
        top_n_genres = st.slider("Hiển thị bao nhiêu thể loại:", 5, 20, 10)
    
    avg_budget = current_data.groupby('genre')['budget'].mean().round()
    if sort_option == "Tăng dần":
        avg_budget = avg_budget.sort_values()
    elif sort_option == "Giảm dần":
        avg_budget = avg_budget.sort_values(ascending=False)
    
    avg_budget = avg_budget.head(top_n_genres).reset_index()
    fig = plt.figure(figsize=(fig_width, fig_height))
    if chart_palette:
        ax = sns.barplot(x=avg_budget['genre'], y=avg_budget['budget'], palette=chart_palette)
    else:
        ax = sns.barplot(x=avg_budget['genre'], y=avg_budget['budget'], color=chart_color)
    
    if show_avg_line:
        avg_value = avg_budget['budget'].mean()
        plt.axhline(y=avg_value, color='red', linestyle='--', label=f'Trung bình: {avg_value:,.0f}')
        plt.legend()
    
    plt.xlabel('Thể loại')
    plt.ylabel('Ngân sách trung bình')
    plt.title('Ngân sách trung bình của phim theo thể loại')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plot_and_save(fig, "budget_by_genre.png")
    
    st.write("### Bảng dữ liệu ngân sách trung bình")
    st.dataframe(avg_budget)
    if st.button("Tải xuống bảng ngân sách (CSV)"):
        csv = avg_budget.to_csv(index=False)
        st.download_button(
            label="Nhấn để tải",
            data=csv,
            file_name="budget_by_genre.csv",
            mime="text/csv"
        )

elif analysis_type == "Phân tích theo năm":
    st.write("## Xu hướng phim theo năm")
    
    with st.sidebar.expander("📅 Phạm vi năm", expanded=True):
        min_year = int(current_data['year'].min())
        max_year = int(current_data['year'].max())
        year_range = st.slider("Chọn khoảng năm:", min_year, max_year, (min_year, max_year))
        year_analysis = st.selectbox(
            "Chọn chỉ số phân tích:",
            ["Số lượng phim", "Điểm đánh giá trung bình", "Ngân sách trung bình", "Doanh thu trung bình"]
        )
    
    filtered_data = current_data[(current_data['year'] >= year_range[0]) & (current_data['year'] <= year_range[1])]
    check_empty_data(filtered_data)
    
    if year_analysis == "Số lượng phim":
        data_by_year = filtered_data.groupby('year').size().reset_index(name='count')
        y_column, y_label, title = 'count', 'Số lượng phim', 'Số lượng phim phát hành theo năm'
    elif year_analysis == "Điểm đánh giá trung bình":
        data_by_year = filtered_data.groupby('year')['score'].mean().reset_index()
        y_column, y_label, title = 'score', 'Điểm đánh giá trung bình', 'Điểm đánh giá phim theo năm'
    elif year_analysis == "Ngân sách trung bình":
        data_by_year = filtered_data.groupby('year')['budget'].mean().reset_index()
        y_column, y_label, title = 'budget', 'Ngân sách trung bình', 'Ngân sách phim theo năm'
    else:
        data_by_year = filtered_data.groupby('year')['gross'].mean().reset_index()
        y_column, y_label, title = 'gross', 'Doanh thu trung bình', 'Doanh thu phim theo năm'
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    if chart_palette:
        sns.lineplot(data=data_by_year, x='year', y=y_column, palette=chart_palette, marker='o')
    else:
        plt.plot(data_by_year['year'], data_by_year[y_column], marker='o', color=chart_color)
    
    plt.xlabel('Năm phát hành')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_and_save(fig, "yearly_analysis.png")
    
    st.write(f"### Dữ liệu {title.lower()}")
    st.dataframe(data_by_year)
    if st.button("Tải xuống dữ liệu năm (CSV)"):
        csv = data_by_year.to_csv(index=False)
        st.download_button(
            label="Nhấn để tải",
            data=csv,
            file_name="yearly_data.csv",
            mime="text/csv"
        )

elif analysis_type == "Top phim":
    st.write("## Top phim theo tiêu chí")
    
    with st.sidebar.expander("🏆 Tùy chọn xếp hạng", expanded=True):
        criteria = st.selectbox("Chọn tiêu chí xếp hạng:", ['score', 'budget', 'gross', 'runtime', 'votes'])
        top_n = st.slider("Số lượng phim hiển thị:", 5, 20, 10)
        chart_type = st.radio("Kiểu biểu đồ:", ["Cột ngang", "Cột đứng", "Bảng"])
        ascending = not st.checkbox("Lấy phim có giá trị cao nhất", True)
    
    valid_data = current_data.dropna(subset=[criteria])
    top_films = valid_data.nsmallest(top_n, criteria) if ascending else valid_data.nlargest(top_n, criteria)
    
    st.write(f"### Top {top_n} phim theo {criteria}")
    if chart_type == "Bảng":
        st.dataframe(top_films[['name', 'genre', 'year', criteria]])
    else:
        fig = plt.figure(figsize=(fig_width, fig_height))
        if chart_type == "Cột ngang":
            if chart_palette:
                sns.barplot(y=top_films['name'], x=top_films[criteria], palette=chart_palette)
            else:
                plt.barh(top_films['name'], top_films[criteria], color=chart_color)
            plt.xlabel(criteria)
            plt.ylabel('Tên phim')
        else:
            if chart_palette:
                sns.barplot(x=top_films['name'], y=top_films[criteria], palette=chart_palette)
            else:
                plt.bar(top_films['name'], top_films[criteria], color=chart_color)
            plt.xlabel('Tên phim')
            plt.ylabel(criteria)
            plt.xticks(rotation=45, ha='right')
        
        plt.title(f'Top {top_n} phim theo {criteria}')
        plt.tight_layout()
        plot_and_save(fig, "top_films.png")
    
    with st.expander("Xem dữ liệu chi tiết"):
        st.dataframe(top_films[['name', 'genre', 'director', 'year', criteria]])
    if st.button("Tải xuống top phim (CSV)"):
        csv = top_films[['name', 'genre', 'director', 'year', criteria]].to_csv(index=False)
        st.download_button(
            label="Nhấn để tải",
            data=csv,
            file_name="top_films.csv",
            mime="text/csv"
        )

elif analysis_type == "Phân tích tương quan":
    st.write("## Phân tích tương quan giữa các yếu tố")
    
    with st.sidebar.expander("🔄 Tùy chọn tương quan", expanded=True):
        numeric_columns = [col for col in current_data.columns if current_data[col].dtype in ['int64', 'float64']]
        x_var = st.selectbox("Chọn biến X:", numeric_columns, index=numeric_columns.index('budget') if 'budget' in numeric_columns else 0)
        y_var = st.selectbox("Chọn biến Y:", numeric_columns, index=numeric_columns.index('gross') if 'gross' in numeric_columns else 1)
        corr_type = st.radio("Kiểu phân tích:", ["Biểu đồ phân tán", "Ma trận tương quan"])
    
    if corr_type == "Biểu đồ phân tán":
        valid_data = current_data.dropna(subset=[x_var, y_var])
        check_empty_data(valid_data)
        correlation = valid_data[x_var].corr(valid_data[y_var])
        fig = plt.figure(figsize=(fig_width, fig_height))
        if chart_palette:
            sns.scatterplot(data=valid_data, x=x_var, y=y_var, hue='genre', palette=chart_palette)
        else:
            plt.scatter(valid_data[x_var], valid_data[y_var], alpha=0.6, color=chart_color)
        
        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.title(f'Mối quan hệ giữa {x_var} và {y_var}')
        plt.annotate(f'Hệ số tương quan: {correlation:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_and_save(fig, "scatter_plot.png")
    
    else:
        corr_vars = st.sidebar.multiselect(
            "Chọn các biến để phân tích:",
            options=numeric_columns,
            default=['budget', 'score', 'gross', 'runtime', 'votes'][:min(5, len(numeric_columns))]
        )
        if corr_vars:
            corr_matrix = current_data[corr_vars].corr()
            fig = plt.figure(figsize=(fig_width, fig_height))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
            plt.title('Ma trận tương quan giữa các biến')
            plt.tight_layout()
            plot_and_save(fig, "correlation_matrix.png")
            
            st.info("""
            *Hướng dẫn đọc ma trận tương quan:*
            - Giá trị gần 1: Tương quan dương mạnh
            - Giá trị gần -1: Tương quan âm mạnh
            - Giá trị gần 0: Ít hoặc không có tương quan
            """)

elif analysis_type == "Phân tích từ khóa":
    st.write("## Phân tích từ khóa trong tiêu đề phim")
    
    with st.sidebar.expander("🔤 Tùy chọn từ khóa", expanded=True):
        min_year = int(current_data['year'].min())
        max_year = int(current_data['year'].max())
        keyword_year_range = st.slider("Lọc phim theo năm:", min_year, max_year, (min_year, max_year))
        max_words = st.slider("Số từ khóa tối đa:", 10, 50, 20)
        keyword_display = st.radio("Kiểu hiển thị:", ["Biểu đồ cột", "Word Cloud"])
    
    year_filtered_data = current_data[(current_data['year'] >= keyword_year_range[0]) & 
                                    (current_data['year'] <= keyword_year_range[1])]
    check_empty_data(year_filtered_data)
    
    all_titles = " ".join(year_filtered_data['name'].dropna().astype(str).str.lower())
    words = re.findall(r'\b\w+\b', all_titles)
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    words = [word for word in words if word not in stopwords and len(word) > 2]
    
    word_counts = Counter(words).most_common(max_words)
    word_df = pd.DataFrame(word_counts, columns=['Từ khóa', 'Số lần xuất hiện'])
    
    if keyword_display == "Biểu đồ cột":
        fig = plt.figure(figsize=(fig_width, fig_height))
        if chart_palette:
            sns.barplot(data=word_df, x='Số lần xuất hiện', y='Từ khóa', palette=chart_palette)
        else:
            sns.barplot(data=word_df, x='Số lần xuất hiện', y='Từ khóa', color=chart_color)
        plt.title(f'Top {max_words} từ khóa phổ biến trong tiêu đề phim')
        plt.xlabel('Số lần xuất hiện')
        plt.ylabel('Từ khóa')
        plt.tight_layout()
        plot_and_save(fig, "keyword_bar.png")
    
    else:
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            max_words=max_words, stopwords=stopwords).generate(all_titles)
        fig = plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud của tiêu đề phim')
        plt.tight_layout()
        plot_and_save(fig, "keyword_wordcloud.png")
    
    st.write("### Bảng từ khóa phổ biến")
    st.dataframe(word_df)
    if st.button("Tải xuống bảng từ khóa (CSV)"):
        csv = word_df.to_csv(index=False)
        st.download_button(
            label="Nhấn để tải",
            data=csv,
            file_name="keywords.csv",
            mime="text/csv"
        )

elif analysis_type == "Thống kê":
    st.write("## Thống kê dữ liệu phim")
    
    with st.sidebar.expander("📊 Tùy chọn thống kê", expanded=True):
        columns_for_stats = st.multiselect(
            "Chọn các cột để xem thống kê:",
            options=[col for col in current_data.columns if current_data[col].dtype in ['int64', 'float64']],
            default=['budget', 'score', 'gross']
        )
        dist_plot_type = st.radio("Kiểu biểu đồ phân phối:", ["Histogram", "Boxplot", "Violin plot"])
    
    if columns_for_stats:
        st.write("### Thống kê mô tả")
        stats_df = current_data[columns_for_stats].describe()
        st.dataframe(stats_df)
        if st.button("Tải xuống thống kê mô tả (CSV)"):
            csv = stats_df.to_csv()
            st.download_button(
                label="Nhấn để tải",
                data=csv,
                file_name="descriptive_stats.csv",
                mime="text/csv"
            )
        
        if st.sidebar.checkbox("Hiển thị biểu đồ phân phối", True):
            col_for_dist = st.sidebar.selectbox("Chọn cột để hiển thị phân phối:", columns_for_stats)
            fig = plt.figure(figsize=(fig_width, fig_height))
            
            if dist_plot_type == "Histogram":
                if chart_palette:
                    sns.histplot(data=current_data, x=col_for_dist, kde=True, palette=chart_palette)
                else:
                    sns.histplot(data=current_data, x=col_for_dist, kde=True, color=chart_color)
                plt.xlabel(col_for_dist)
                plt.ylabel('Tần suất')
                plt.title(f'Phân phối của {col_for_dist}')
            elif dist_plot_type == "Boxplot":
                if chart_palette:
                    sns.boxplot(data=current_data, y=col_for_dist, palette=chart_palette)
                else:
                    sns.boxplot(data=current_data, y=col_for_dist, color=chart_color)
                plt.ylabel(col_for_dist)
                plt.title(f'Boxplot của {col_for_dist}')
            else:
                if chart_palette:
                    sns.violinplot(data=current_data, y=col_for_dist, palette=chart_palette)
                else:
                    sns.violinplot(data=current_data, y=col_for_dist, color=chart_color)
                plt.ylabel(col_for_dist)
                plt.title(f'Violin plot của {col_for_dist}')
            
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plot_and_save(fig, "distribution_plot.png")
            
            with st.expander("Thống kê nâng cao"):
                skewness = current_data[col_for_dist].skew()
                kurtosis = current_data[col_for_dist].kurtosis()
                st.write(f"*Độ xiên (Skewness):* {skewness:.3f}")
                st.write(f"*Độ nhọn (Kurtosis):* {kurtosis:.3f}")
                if skewness > 1:
                    st.info("Phân phối bị xiên phải mạnh")
                elif skewness > 0.5:
                    st.info("Phân phối bị xiên phải vừa phải")
                elif skewness < -1:
                    st.info("Phân phối bị xiên trái mạnh")
                elif skewness < -0.5:
                    st.info("Phân phối bị xiên trái vừa phải")
                else:
                    st.info("Phân phối tương đối cân đối")
# Nguyen Gia Minh 2321050064, Chu Quang Khải 2321050004