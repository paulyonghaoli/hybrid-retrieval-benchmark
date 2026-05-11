$topics = @(
    'supervised learning',
    'unsupervised learning',
    'reinforcement learning',
    'neural networks',
    'decision trees',
    'support vector machines',
    'clustering',
    'feature engineering',
    'model evaluation',
    'natural language processing'
)

$templates = @(
    '{0} is a foundational machine learning approach that {1}. It is often used with datasets that {2}.',
    'A key aspect of {0} is the way it {1}. Practitioners often combine it with {2}.',
    'The study of {0} includes methods like {2} and {1}. This helps models generalize when {3}.'
)

$details = @(
    'learns from labelled examples',
    'discovers clusters in data',
    'optimizes a reward objective',
    'uses backpropagation to adjust weights',
    'reduces overfitting with regularization',
    'measures distance between points',
    'identifies high-variance features',
    'estimates probability distributions',
    'builds hierarchical decision rules'
)

$examples = @(
    'large-scale training sets',
    'complex feature representations',
    'text embeddings for semantic search',
    'cross-validation for robust evaluation',
    'graph data or sequential signals',
    'dimensionality reduction first',
    'ensemble methods for prediction',
    'online learning from streaming inputs'
)

$scenarios = @(
    'labels are sparse',
    'the dataset contains noise',
    'the model must generalize to new data',
    'the feature space is high dimensional',
    'the task requires ranking documents'
)

$queryTypes = @('term_match', 'semantic', 'balanced')
$queryTemplates = @(
    'Which algorithm is best for {0}?',
    'How does {0} differ from {1}?',
    'Explain {0} in simple terms.',
    'What are practical examples of {0}?',
    'When should you choose {0} over {1}?'
)

$basePath = Split-Path -Parent $MyInvocation.MyCommand.Path
$corpusFile = Join-Path $basePath 'corpus.jsonl'
$queryFile = Join-Path $basePath 'queries.jsonl'

if (Test-Path $corpusFile) { Remove-Item $corpusFile }
if (Test-Path $queryFile) { Remove-Item $queryFile }

$docs = 1..200 | ForEach-Object {
    $topic = $topics[($_ - 1) % $topics.Count]
    $detail = Get-Random -InputObject $details
    $example = Get-Random -InputObject $examples
    $scenario = Get-Random -InputObject $scenarios
    $template = Get-Random -InputObject $templates
    $list = Get-Random -InputObject $topics -Count 2
    $text = [string]::Format($template, $topic, $detail, $example, $scenario) + ' The document also discusses ' + ($list -join ', ') + '.'
    [PSCustomObject]@{
        doc_id = ('doc_{0:000}' -f $_)
        title = 'Document on ' + $topic
        text = $text
    }
}

$docs | ForEach-Object { $_ | ConvertTo-Json -Compress | Out-File -FilePath $corpusFile -Append -Encoding utf8 }

0..29 | ForEach-Object {
    $index = $_
    $queryType = $queryTypes[$index % $queryTypes.Count]
    $topic = Get-Random -InputObject $topics
    $related = Get-Random -InputObject ($topics | Where-Object { $_ -ne $topic })
    $template = Get-Random -InputObject $queryTemplates
    $queryText = [string]::Format($template, $topic, $related)

    if ($queryType -eq 'term_match') {
        $relevant = $docs | Where-Object { $_.title -like "*${topic}*" }
    } elseif ($queryType -eq 'semantic') {
        $keyword = $topic.Split(' ')[0]
        $relevant = $docs | Where-Object { $_.text -match 'semantic|similarity|concept' -or $_.text -like "*${keyword}*" }
    } else {
        $relevant = $docs | Where-Object { $_.text -like "*${topic}*" -or $_.text -like "*${related}*" }
    }

    if (-not $relevant) {
        $relevant = $docs[0..0]
    }

    $relevantIds = @($relevant[0..([Math]::Min(5, $relevant.Count - 1))] | ForEach-Object { $_.doc_id })
    $queryId = 'q_' + ($index + 1).ToString('D2')
    [PSCustomObject]@{
        query_id = $queryId
        query = $queryText
        relevant_doc_ids = $relevantIds
        query_type = $queryType
    } | ConvertTo-Json -Compress | Out-File -FilePath $queryFile -Append -Encoding utf8
}

Write-Host "Generated corpus and queries at $basePath"
