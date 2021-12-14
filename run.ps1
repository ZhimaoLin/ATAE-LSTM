$glovePathList = @('./glove_embedding/glove.twitter.27B.25d.txt', 
                    './glove_embedding/glove.twitter.27B.200d.txt',
                    './glove_embedding/glove.6B.200d.txt',
                    './glove_embedding/glove.6B.300d.txt')
                    # './glove_embedding/glove.42B.300d.txt')

$wordEmbeddingDimList = @(25, 200, 200, 300)
$batchSizeList = @(5, 10)
$epochList = @(5, 10)
$hiddenDimList = $wordEmbeddingDimList



for ($i = 0; $i -lt $glovePathList.Length; $i++) {
    $glovePath = $glovePathList[$i]
    $wordEmbeddingDim = $wordEmbeddingDimList[$i]
    $hiddenDim = $hiddenDimList[$i]

    foreach ($batchSize in $batchSizeList) {
        foreach ($epoch in $epochList) {

            Write-Host "Start" -ForegroundColor Yellow
            Write-Host("=================================")
            Write-Host("Configurations are the following: ")
            Write-Host("=================================")
            Write-Host("Glove path: $($glovePath)")
            Write-Host("Word embedding dimension: $($wordEmbeddingDim)")
            Write-Host("Batch size: $($batchSize)")
            Write-Host("Epoch: $($epoch)")
            Write-Host("Hidden dimension: $($hiddenDim)")

            Write-Host "Running" -ForegroundColor Yellow

            $gloveName = $glovePath.split("/")[2]
            $resultName = "$($gloveName)_WE$($wordEmbeddingDim)_BS$($batchSize)_EP$($epoch)_HD$($hiddenDim).txt"
            python ./train_and_test.py --data_path=./data/covid_dataset.csv --glove_path=$glovePath --batch_size=$batchSize --epoch=$epoch --word_embedding_dim=$wordEmbeddingDim --hidden_dim=$hiddenDim > result/$resultName

            Write-Host "Done" -ForegroundColor Green
            Write-Host "`n`n"     
        }
    }

}


# python ./train_and_test.py --data_path=./data/covid_dataset.csv --glove_path=./glove_embedding/glove.twitter.27B.25d.txt --batch_size=10 --epoch=5 --word_embedding_dim=25 --hidden_dim=32 > result/1.txt