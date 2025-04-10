$(document).ready(function() {
    // 系统状态变量
    let systemRunning = false;
    let circuitCreated = false;
    let circuitExecuted = false;
    let marketDataConverted = false;
    let resultsInterpreted = false;
    let marketAnalyzed = false;
    
    // 图表实例
    let stockChart = null;
    let quantumStateChart = null;
    
    // 系统控制
    $('#startSystem').click(function() {
        $(this).prop('disabled', true);
        $(this).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 启动中...');
        
        $.ajax({
            url: '/api/system/start',
            type: 'POST',
            success: function(response) {
                systemRunning = true;
                updateSystemStatus(true);
                $('#startSystem').html('启动系统');
                $('#startSystem').prop('disabled', true);
                $('#stopSystem').prop('disabled', false);
                
                // 启用功能按钮
                $('#createCircuit').prop('disabled', false);
                $('#convertMarketData').prop('disabled', false);
                $('#convertMethod').prop('disabled', false);
                $('#showStockData').prop('disabled', false);
                $('#analyzeMarket').prop('disabled', false);
                
                updateComponentStatus(response.components);
                showNotification('系统启动成功', 'success');
            },
            error: function(error) {
                $('#startSystem').html('启动系统');
                $('#startSystem').prop('disabled', false);
                showNotification('系统启动失败: ' + error.responseJSON?.error || '未知错误', 'danger');
            }
        });
    });
    
    $('#stopSystem').click(function() {
        $(this).prop('disabled', true);
        $(this).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 停止中...');
        
        $.ajax({
            url: '/api/system/stop',
            type: 'POST',
            success: function(response) {
                systemRunning = false;
                updateSystemStatus(false);
                $('#stopSystem').html('停止系统');
                $('#stopSystem').prop('disabled', true);
                $('#startSystem').prop('disabled', false);
                
                // 禁用功能按钮
                $('#createCircuit').prop('disabled', true);
                $('#executeCircuit').prop('disabled', true);
                $('#convertMarketData').prop('disabled', true);
                $('#convertMethod').prop('disabled', true);
                $('#interpretResults').prop('disabled', true);
                $('#interpretMethod').prop('disabled', true);
                $('#showStockData').prop('disabled', true);
                $('#analyzeMarket').prop('disabled', true);
                
                // 重置状态
                circuitCreated = false;
                circuitExecuted = false;
                marketDataConverted = false;
                resultsInterpreted = false;
                marketAnalyzed = false;
                
                updateComponentStatus({
                    backend: { status: 'stopped' },
                    converter: { status: 'stopped' },
                    interpreter: { status: 'stopped' },
                    analyzer: { status: 'stopped' }
                });
                
                showNotification('系统已停止', 'info');
            },
            error: function(error) {
                $('#stopSystem').html('停止系统');
                $('#stopSystem').prop('disabled', false);
                showNotification('系统停止失败: ' + error.responseJSON?.error || '未知错误', 'danger');
            }
        });
    });
    
    // 量子电路操作
    $('#createCircuit').click(function() {
        $(this).prop('disabled', true);
        $(this).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 创建中...');
        
        $.ajax({
            url: '/api/quantum/create_circuit',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ circuit_type: 'bell', qubits: 2 }),
            success: function(response) {
                circuitCreated = true;
                $('#createCircuit').html('创建Bell状态电路');
                $('#createCircuit').prop('disabled', true);
                $('#executeCircuit').prop('disabled', false);
                
                $('#circuitInfo').html(`
                    <div class="alert alert-success">
                        <h5>电路已创建</h5>
                        <p>类型: Bell状态电路</p>
                        <p>描述: 创建了2个量子比特的纠缠态</p>
                        <p>门操作: Hadamard门 (H) 在第一个量子比特上, 然后是两个量子比特之间的CNOT门</p>
                    </div>
                `);
                
                showNotification('量子电路创建成功', 'success');
            },
            error: function(error) {
                $('#createCircuit').html('创建Bell状态电路');
                $('#createCircuit').prop('disabled', false);
                showNotification('电路创建失败: ' + error.responseJSON?.error || '未知错误', 'danger');
            }
        });
    });
    
    $('#executeCircuit').click(function() {
        $(this).prop('disabled', true);
        $(this).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 执行中...');
        
        $.ajax({
            url: '/api/quantum/execute_circuit',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ shots: 1000 }),
            success: function(response) {
                circuitExecuted = true;
                $('#executeCircuit').html('执行电路');
                $('#executeCircuit').prop('disabled', true);
                $('#interpretResults').prop('disabled', false);
                $('#interpretMethod').prop('disabled', false);
                
                // 显示量子状态结果
                $('#circuitResults').show();
                displayQuantumResults(response.results);
                
                showNotification('量子电路执行成功', 'success');
            },
            error: function(error) {
                $('#executeCircuit').html('执行电路');
                $('#executeCircuit').prop('disabled', false);
                showNotification('电路执行失败: ' + error.responseJSON?.error || '未知错误', 'danger');
            }
        });
    });
    
    // 量子解释
    $('#interpretResults').click(function() {
        $(this).prop('disabled', true);
        $(this).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 解释中...');
        
        const method = $('#interpretMethod').val();
        
        $.ajax({
            url: '/api/quantum/interpret',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ method: method }),
            success: function(response) {
                resultsInterpreted = true;
                $('#interpretResults').html('解释量子结果');
                $('#interpretResults').prop('disabled', false);
                
                // 显示解释结果
                displayInterpretation(response, method);
                
                showNotification('量子结果解释成功', 'success');
            },
            error: function(error) {
                $('#interpretResults').html('解释量子结果');
                $('#interpretResults').prop('disabled', false);
                showNotification('结果解释失败: ' + error.responseJSON?.error || '未知错误', 'danger');
            }
        });
    });
    
    // 市场数据转换
    $('#convertMarketData').click(function() {
        $(this).prop('disabled', true);
        $(this).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 转换中...');
        
        const method = $('#convertMethod').val();
        
        $.ajax({
            url: '/api/market/convert',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ encoding_method: method }),
            success: function(response) {
                marketDataConverted = true;
                $('#convertMarketData').html('转换市场数据');
                $('#convertMarketData').prop('disabled', false);
                
                // 显示转换结果
                $('#conversionResults').show();
                $('#conversionSummary').html(`
                    <div class="alert alert-success">
                        <h5>转换完成</h5>
                        <p>编码方法: ${method === 'amplitude' ? '振幅编码' : '角度编码'}</p>
                        <p>处理的数据点: ${response.processed_datapoints}</p>
                        <p>量子态维度: ${response.quantum_dimension}</p>
                        <p>归一化因子: ${response.normalization_factor.toFixed(4)}</p>
                    </div>
                `);
                
                showNotification('市场数据转换成功', 'success');
            },
            error: function(error) {
                $('#convertMarketData').html('转换市场数据');
                $('#convertMarketData').prop('disabled', false);
                showNotification('数据转换失败: ' + error.responseJSON?.error || '未知错误', 'danger');
            }
        });
    });
    
    // 市场分析
    $('#analyzeMarket').click(function() {
        $(this).prop('disabled', true);
        $(this).html('<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> 分析中...');
        
        $.ajax({
            url: '/api/market/analyze',
            type: 'POST',
            success: function(response) {
                marketAnalyzed = true;
                $('#analyzeMarket').html('分析市场数据');
                $('#analyzeMarket').prop('disabled', false);
                
                // 显示分析结果
                displayMarketAnalysis(response);
                
                showNotification('市场分析完成', 'success');
            },
            error: function(error) {
                $('#analyzeMarket').html('分析市场数据');
                $('#analyzeMarket').prop('disabled', false);
                showNotification('市场分析失败: ' + error.responseJSON?.error || '未知错误', 'danger');
            }
        });
    });
    
    // 显示市场数据
    $('#showStockData').click(function() {
        $.ajax({
            url: '/api/market/data',
            type: 'GET',
            success: function(response) {
                initializeStockChart(response);
                showNotification('市场数据加载成功', 'success');
            },
            error: function(error) {
                showNotification('市场数据加载失败: ' + error.responseJSON?.error || '未知错误', 'danger');
            }
        });
    });
    
    // 辅助函数
    function updateSystemStatus(isRunning) {
        if (isRunning) {
            $('#systemStatusIndicator .system-status')
                .removeClass('status-stopped')
                .addClass('status-running');
            $('#systemStatusText').text('系统运行中');
        } else {
            $('#systemStatusIndicator .system-status')
                .removeClass('status-running')
                .addClass('status-stopped');
            $('#systemStatusText').text('系统未启动');
        }
    }
    
    function updateComponentStatus(components) {
        // 更新量子后端状态
        updateComponentStatusRow('量子后端', components.backend);
        
        // 更新市场到量子转换器状态
        updateComponentStatusRow('市场到量子转换器', components.converter);
        
        // 更新量子解释器状态
        updateComponentStatusRow('量子解释器', components.interpreter);
        
        // 更新市场分析器状态
        updateComponentStatusRow('市场分析器', components.analyzer);
    }
    
    function updateComponentStatusRow(componentName, status) {
        const $row = $('#componentStatus tr').filter(function() {
            return $(this).find('td:first').text() === componentName;
        });
        
        const $statusCell = $row.find('td:last');
        $statusCell.html('');
        
        if (status === 'running') {
            $statusCell.html('<span class="system-status status-running"></span> 运行中');
        } else {
            $statusCell.html('<span class="system-status status-stopped"></span> 未启动');
        }
    }
    
    function displayQuantumResults(results) {
        // 显示量子状态结果
        const $resultsContainer = $('#quantumStateResults');
        $resultsContainer.empty();
        
        const states = Object.keys(results);
        const stateClasses = ['state-00', 'state-01', 'state-10', 'state-11'];
        
        states.forEach((state, index) => {
            const probability = results[state];
            const className = stateClasses[index % stateClasses.length];
            
            $resultsContainer.append(`
                <div class="quantum-state ${className}">
                    |${state}⟩: ${(probability * 100).toFixed(1)}%
                </div>
            `);
        });
        
        // 初始化量子态图表
        initializeQuantumStateChart(results);
    }
    
    function initializeQuantumStateChart(results) {
        const states = Object.keys(results);
        const probabilities = states.map(state => results[state] * 100);
        
        const options = {
            series: [{
                name: '概率',
                data: probabilities
            }],
            chart: {
                type: 'bar',
                height: 250,
                toolbar: {
                    show: false
                }
            },
            plotOptions: {
                bar: {
                    borderRadius: 4,
                    horizontal: false,
                    columnWidth: '55%',
                    distributed: true
                }
            },
            dataLabels: {
                enabled: true,
                formatter: function(val) {
                    return val.toFixed(1) + '%';
                },
                style: {
                    fontSize: '12px',
                    colors: ['#fff']
                }
            },
            colors: ['#2196F3', '#FF9800', '#9C27B0', '#4CAF50'],
            xaxis: {
                categories: states.map(state => `|${state}⟩`),
                labels: {
                    style: {
                        fontSize: '12px'
                    }
                }
            },
            yaxis: {
                max: 100,
                title: {
                    text: '概率 (%)'
                }
            },
            title: {
                text: '量子态概率分布',
                align: 'center',
                style: {
                    fontSize: '16px'
                }
            }
        };
        
        if (quantumStateChart) {
            quantumStateChart.destroy();
        }
        
        quantumStateChart = new ApexCharts(document.querySelector("#quantumStateChart"), options);
        quantumStateChart.render();
    }
    
    function displayInterpretation(response, method) {
        $('#interpretationResults').show();
        
        // 更新信号
        const signal = response.signal;
        $('#quantumSignal')
            .removeClass()
            .addClass('quantum-signal')
            .addClass(`signal-${signal.toLowerCase().replace(' ', '_')}`)
            .text(getSignalText(signal));
        
        // 更新信号强度
        const strength = response.strength * 100;
        $('#strengthValue')
            .css('width', `${strength}%`)
            .css('background-color', getSignalColor(signal))
            .text(`${strength.toFixed(1)}%`);
        
        // 更新概率信息
        $('#upProbability').text(`${(response.up_probability * 100).toFixed(1)}%`);
        $('#downProbability').text(`${(response.down_probability * 100).toFixed(1)}%`);
        
        // 更新状态信息
        $('#mostProbableState').text(`|${response.most_probable_state}⟩`);
        $('#interpretationMethod').text(method === 'probability' ? '概率解释' : '阈值解释');
    }
    
    function getSignalText(signal) {
        switch(signal) {
            case 'STRONG_BUY': return '强烈买入信号';
            case 'BUY': return '买入信号';
            case 'NEUTRAL': return '中性信号';
            case 'SELL': return '卖出信号';
            case 'STRONG_SELL': return '强烈卖出信号';
            default: return signal;
        }
    }
    
    function getSignalColor(signal) {
        switch(signal) {
            case 'STRONG_BUY': return '#388e3c';
            case 'BUY': return '#8bc34a';
            case 'NEUTRAL': return '#9e9e9e';
            case 'SELL': return '#ff9800';
            case 'STRONG_SELL': return '#d32f2f';
            default: return '#9e9e9e';
        }
    }
    
    function initializeStockChart(data) {
        const stockData = data.prices;
        const dates = stockData.map(item => item.date);
        const prices = stockData.map(item => item.close);
        
        const options = {
            series: [{
                name: data.symbol,
                data: prices
            }],
            chart: {
                type: 'area',
                height: 400,
                zoom: {
                    type: 'x',
                    enabled: true,
                    autoScaleYaxis: true
                },
                toolbar: {
                    autoSelected: 'zoom'
                }
            },
            dataLabels: {
                enabled: false
            },
            stroke: {
                curve: 'straight',
                width: 2
            },
            colors: ['#4527a0'],
            fill: {
                type: 'gradient',
                gradient: {
                    shadeIntensity: 1,
                    opacityFrom: 0.7,
                    opacityTo: 0.3,
                    stops: [0, 100]
                }
            },
            markers: {
                size: 0
            },
            xaxis: {
                type: 'datetime',
                categories: dates,
                labels: {
                    formatter: function(value) {
                        return new Date(value).toLocaleDateString();
                    }
                }
            },
            yaxis: {
                title: {
                    text: '价格'
                },
                labels: {
                    formatter: function(val) {
                        return val.toFixed(2);
                    }
                }
            },
            tooltip: {
                shared: false,
                x: {
                    formatter: function(val) {
                        return new Date(val).toLocaleDateString();
                    }
                },
                y: {
                    formatter: function(val) {
                        return val.toFixed(2);
                    }
                },
                theme: 'dark'
            },
            title: {
                text: `${data.symbol} 历史价格`,
                align: 'center',
                style: {
                    fontSize: '16px'
                }
            }
        };

        if (stockChart) {
            stockChart.destroy();
        }
        
        stockChart = new ApexCharts(document.querySelector("#stockChart"), options);
        stockChart.render();
    }
    
    function displayMarketAnalysis(response) {
        $('#marketAnalysisResults').show();
        
        const $tableBody = $('#analysisTable');
        $tableBody.empty();
        
        response.stocks.forEach(stock => {
            const trendDirection = getTrendDirectionText(stock.trend_direction);
            const trendStrength = (stock.trend_strength * 100).toFixed(1) + '%';
            const volatility = (stock.volatility * 100).toFixed(2) + '%';
            
            $tableBody.append(`
                <tr>
                    <td>${stock.symbol}</td>
                    <td><strong>${stock.score.toFixed(2)}</strong></td>
                    <td>${stock.price.toFixed(2)}</td>
                    <td class="text-${getTrendColor(stock.trend_direction)}">${trendDirection}</td>
                    <td>${trendStrength}</td>
                    <td>${volatility}</td>
                </tr>
            `);
        });
    }
    
    function getTrendDirectionText(trend) {
        switch(trend) {
            case 'UP': return '上升';
            case 'DOWN': return '下降';
            case 'SIDEWAYS': return '盘整';
            default: return trend;
        }
    }
    
    function getTrendColor(trend) {
        switch(trend) {
            case 'UP': return 'success';
            case 'DOWN': return 'danger';
            case 'SIDEWAYS': return 'warning';
            default: return 'secondary';
        }
    }
    
    function showNotification(message, type) {
        // 简单的通知实现，实际应用中可以使用toast组件
        console.log(`[${type.toUpperCase()}] ${message}`);
        // 这里可以添加更复杂的通知实现
    }
    
    // 初始化状态检查
    function checkSystemStatus() {
        $.ajax({
            url: '/api/system/status',
            type: 'GET',
            success: function(response) {
                if (response.components && (response.components.backend === 'running' || 
                    response.components.converter === 'running' || 
                    response.components.interpreter === 'running' || 
                    response.components.analyzer === 'running')) {
                    systemRunning = true;
                    updateSystemStatus(true);
                    $('#startSystem').prop('disabled', true);
                    $('#stopSystem').prop('disabled', false);
                    
                    // 启用功能按钮
                    $('#createCircuit').prop('disabled', false);
                    $('#convertMarketData').prop('disabled', false);
                    $('#convertMethod').prop('disabled', false);
                    $('#showStockData').prop('disabled', false);
                    $('#analyzeMarket').prop('disabled', false);
                    
                    updateComponentStatus(response.components);
                }
            }
        });
    }
    
    // 页面加载时检查系统状态
    checkSystemStatus();
}); 