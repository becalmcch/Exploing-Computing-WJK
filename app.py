import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="조선 3사 주가 분석 및 예측", layout="wide")

st.title("국내 조선 3사 주가 분석 및 AI 예측")
st.markdown("""
1. 상관관계 분석 (Correlation): 3사 주가의 동조화 현상 분석
2. 미래 주가 예측: 향후 30일간의 주가 추이 예측
""")

# 데이터 로드
try:
    df = pd.read_csv('ship_bigdata.csv')
    df['Date'] = pd.to_datetime(df['Date'])
except:
    st.error("'ship_bigdata.csv' 파일이 없습니다.")
    st.stop()

# 탭 구성
tab1, tab2 = st.tabs(["📊 주가 상관관계 분석", "🤖 AI 주가 예측 (LSTM)"])

# --- 탭 1: 상관관계 분석 ---
with tab1:
    st.subheader("1. 조선 3사 주가 변동성 비교")
    
    # 과거 데이터만 필터링
    history_df = df[df['Type'] == 'History']
    
    # 1) 전체 추이 그래프
    fig_line = px.line(history_df, x='Date', y='Price', color='Company', 
                       title="최근 3년 주가 변동 추이")
    st.plotly_chart(fig_line, use_container_width=True)
    
    # 2) 상관관계 히트맵
    st.subheader("2. 기업 간 주가 상관관계 (Correlation Heatmap)")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("""
        분석 포인트:
        * 색이 붉을수록(1에 가까울수록) 두 기업의 주가가 비슷하게 움직입니다.
        * 조선업은 업황의 영향을 크게 받으므로 매우 높은 상관관계를 보입니다.
        """)
        
    with col2:
        # 피벗 후 상관계수 계산
        pivot_df = history_df.pivot(index='Date', columns='Company', values='Price')
        corr_matrix = pivot_df.corr()
        
        fig_corr = px.imshow(corr_matrix, text_auto=True, 
                             color_continuous_scale='RdBu_r', aspect='auto')
        st.plotly_chart(fig_corr, use_container_width=True)

# --- 탭 2: AI 예측 ---
with tab2:
    st.subheader("🧠 LSTM 딥러닝 기반 미래 주가 예측")
    st.write("학습된 LSTM 모델이 예측한 향후 30일(1개월)의 주가 흐름입니다.")
    
    # 회사 선택
    selected_company = st.selectbox("확인할 기업을 선택하세요", df['Company'].unique())
    
    # 해당 회사 데이터 필터링
    comp_data = df[df['Company'] == selected_company]
    history_data = comp_data[comp_data['Type'] == 'History']
    pred_data = comp_data[comp_data['Type'] == 'Prediction']
    
    # 그래프 그리기
    fig_pred = go.Figure()
    
    # 과거 데이터 (실선)
    fig_pred.add_trace(go.Scatter(
        x=history_data['Date'], y=history_data['Price'],
        mode='lines', name='실제 주가 (History)',
        line=dict(color='royalblue')
    ))
    
    # 미래 예측 (점선 + 빨간색) - 이어지게 하기 위해 과거 마지막 점 추가
    last_hist = history_data.iloc[-1]
    pred_x = [last_hist['Date']] + pred_data['Date'].tolist()
    pred_y = [last_hist['Price']] + pred_data['Predicted_Price'].tolist()
    
    fig_pred.add_trace(go.Scatter(
        x=pred_x, y=pred_y,
        mode='lines+markers', name='AI 예측 (Prediction)',
        line=dict(color='red', dash='dot', width=3)
    ))
    
    fig_pred.update_layout(title=f"{selected_company} 주가 예측 시뮬레이션", 
                           xaxis_title="날짜", yaxis_title="주가(원)")
    st.plotly_chart(fig_pred, use_container_width=True)
    
    st.success(f"✅ 분석 결과: LSTM 모델은 현재의 추세를 반영하여 {selected_company}의 단기 변동성을 위와 같이 예측했습니다.")